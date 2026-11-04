"""Collect session-separated schema-v2 labels while vision continues in a worker."""
import argparse
import csv
import math
import time
import uuid
from pathlib import Path
from vision.cli import ROOT, add_vision_arguments, make_pipeline
from vision.features import FEATURES, SCHEMA_ID, feature_row
from vision.service import VisionService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_vision_arguments(parser)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/focus_v2.csv')
    args = parser.parse_args()
    if args.preview:
        parser.error('Collection uses terminal prompts; omit --preview (use live_inference.py for video preview)')
    fields = ['session_id', 'schema_id', 'landmark_schema', 'calibration_id', 'timestamp', *FEATURES, 'focus_score']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size:
        with args.output.open(newline='') as source:
            if next(csv.reader(source)) != fields:
                parser.error('Output has an incompatible schema; choose a new file instead of mixing v1/v2 rows')
    pipeline = service = desktop = None
    try:
        pipeline = make_pipeline(args)
        from desktop_pipeline import DesktopPipeline
        desktop = DesktopPipeline()
        service = VisionService(pipeline, args.camera, args.width, args.height, args.fps)
        service.start()
        session_id = uuid.uuid4().hex
        next_prompt = time.monotonic() + 60
        print(f'Collecting session {session_id}, actual backend {pipeline.actual_backend}. Labels refer to the minute ending at each prompt. Ctrl+C to stop.')
        with args.output.open('a', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            if output.tell() == 0:
                writer.writeheader()
            while True:
                result = service.get()
                if result is None:
                    if service.results.closed:
                        break
                    continue
                metric, _ = result
                now = time.monotonic()
                if now < next_prompt:
                    continue
                next_prompt = now + 60
                row = feature_row(metric, desktop.get_current_metrics()) if now - metric.timestamp <= 0.2 else None
                if row is None:
                    print(f'Skipping label: insufficient fresh eye data (coverage={metric.valid_coverage_60s:.0%}).')
                    continue
                # Freeze features at the prompt; the vision service continues during input.
                text = input('Focus in the minute just ended, 0–100 (Enter skips, q quits): ').strip()
                if text.lower() == 'q':
                    break
                try:
                    score = float(text)
                    if not math.isfinite(score) or not 0 <= score <= 100:
                        raise ValueError()
                except ValueError:
                    print('Skipped; enter a finite number from 0 to 100.')
                    continue
                writer.writerow(dict(session_id=session_id, schema_id=SCHEMA_ID,
                                     landmark_schema=pipeline.identity['schema'], calibration_id=metric.calibration_id,
                                     timestamp=metric.timestamp, **row, focus_score=score))
                output.flush()
                print(f'Saved to {args.output}')
                # Avoid immediately prompting again after a long input delay.
                next_prompt = time.monotonic() + 60
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if service:
                service.close()
            elif pipeline:
                pipeline.close()
        finally:
            if desktop:
                desktop.close()


if __name__ == '__main__':
    main()

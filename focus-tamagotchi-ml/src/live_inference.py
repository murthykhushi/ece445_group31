"""Local vision service with optional focus prediction and STM32 output."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from vision.cli import add_vision_arguments, make_pipeline, ROOT
from vision.features import feature_row, load_focus_model, predict_focus
from vision.service import VisionService
from vision.telemetry import Telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_vision_arguments(parser)
    parser.add_argument('--vision-only', action='store_true', help='Run metrics without desktop hooks or a trained focus model')
    parser.add_argument('--focus-model', type=Path, default=ROOT / 'focus_v2.joblib')
    parser.add_argument('--duration', type=float, default=0, help='Run seconds; zero runs until Ctrl+C')
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--warmup', type=float, default=30)
    parser.add_argument('--serial-port', help='Optional COM3 or /dev/ttyACM0, etc.')
    parser.add_argument('--serial-protocol', choices=['json-v2', 'legacy'], default='json-v2')
    args = parser.parse_args()
    if args.duration < 0 or args.warmup < 0:
        parser.error('duration and warmup must be nonnegative')
    if args.vision_only and args.serial_port:
        parser.error('--serial-port requires focus prediction, not --vision-only')
    service = telemetry = desktop = serial_output = pipeline = None
    try:
        bundle = None if args.vision_only else load_focus_model(args.focus_model)
        pipeline = make_pipeline(args)
        if bundle is not None and pipeline.identity['schema'] not in bundle.get('landmark_schemas', []):
            raise ValueError('Focus model has not been trained with this landmark schema')
        run_dir = args.log_dir or ROOT / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        telemetry = Telemetry(run_dir, pipeline.identity, args.warmup)
        service = VisionService(pipeline, args.camera, args.width, args.height, args.fps, args.preview)
        service.telemetry = telemetry
        if not args.vision_only:
            from desktop_pipeline import DesktopPipeline
            desktop = DesktopPipeline()
        if args.serial_port:
            from vision.output import SerialOutput
            serial_output = SerialOutput(args.serial_port, args.serial_protocol)
            if args.serial_protocol == 'legacy':
                print('Legacy firmware cannot express UNKNOWN/TTL; on missing data it retains the last state. Use json-v2 with matching firmware for expiry.')
        service.start()
        telemetry.write_manifest(dict(camera=service.capture.manifest, arguments=vars(args)))
        print(f'Actual backend: {pipeline.actual_backend}; fallback: {pipeline.fallback_reason}; logs: {run_dir}')
        print('Ctrl+C to stop. Preview is ' + ('enabled (q to quit).' if args.preview else 'disabled.'))
        start = time.monotonic()
        last_log = last_send = 0.0
        next_prediction = start + 60
        score, score_at = None, 0.0
        while not args.duration or time.monotonic() - start < args.duration:
            result = service.get()
            now = time.monotonic()
            if result is None:
                score = None
                if serial_output:
                    serial_output.send('UNKNOWN')
                if service.results.closed:
                    break
                continue
            metric, image = result
            recent = now - metric.timestamp <= 0.2 and 'stale_frame' not in metric.quality_flags
            usable = recent and metric.eye_state != 'UNKNOWN' and metric.closure_fraction_60s is not None
            if not usable:
                score = None
            if desktop and now >= next_prediction:
                row = feature_row(metric, desktop.get_current_metrics()) if recent else None
                if row is not None:
                    raw = predict_focus(bundle, row)
                    score = raw if score is None else 0.3 * raw + 0.7 * score
                    score_at = now
                else:
                    score = None
                next_prediction = now + 60
            if now - score_at > 65:
                score = None
            state = 'UNKNOWN' if score is None else ('FOCUSED' if score >= 50 else 'UNFOCUSED')
            if serial_output and now-last_send >= 1:
                serial_output.send(state, score)
                last_send = now
                if serial_output.error:
                    raise RuntimeError('Serial output failed: ' + serial_output.error)
            if now-last_log >= 1:
                closure = 'unknown' if metric.closure_fraction_60s is None else f'{metric.closure_fraction_60s:.3f}'
                print(f'{metric.backend_actual} eye={metric.eye_state} closure={closure} coverage={metric.valid_coverage_60s:.0%} age={metric.frame_age_ms:.1f}ms focus={state} flags={",".join(metric.quality_flags)}')
                last_log = now
            if args.preview:
                import cv2
                preview = image.copy()
                cv2.putText(preview, f'{metric.backend_actual} | {metric.eye_state} | {state}', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 150), 1)
                cv2.imshow('Focus vision v2', preview)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if service:
                service.close()
            elif pipeline:
                pipeline.close()
        finally:
            if telemetry:
                telemetry.close()
            if desktop:
                desktop.close()
            if serial_output:
                serial_output.close()
            if args.preview:
                import cv2
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

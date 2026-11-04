"""Compare run directories, preserving backend identity and benchmark limitations."""
import argparse
import json
from pathlib import Path
import pandas as pd


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    df = pd.read_csv(directory / 'telemetry.csv')
    if df.empty:
        raise ValueError(f'No telemetry samples in {directory}')
    df = df[~df.warmup.astype(str).str.lower().eq('true')]
    if df.empty:
        raise ValueError(f'No post-warmup samples in {directory}; run longer or explicitly set --warmup 0 for a smoke test')
    cpu = df.cpu_machine_percent
    durations = df['interval_s'] if 'interval_s' in df else pd.Series(1.0, index=df.index)
    cpu_mean = float((cpu * durations).sum() / durations.sum())
    result = dict(run=str(directory), backend=manifest['identity']['backend'], profile=manifest['identity'].get('profile'),
                  fallback=manifest['identity'].get('fallback_reason'), cpu_machine_mean=cpu_mean,
                  cpu_machine_p95=float(cpu.quantile(0.95)), cpu_machine_max=float(cpu.max()),
                  fraction_cpu_intervals_ge_10=float((cpu >= 10).mean()),
                  mean_completed_fps=float((df.completed_fps * durations).sum() / durations.sum()), frame_age_max_ms=float(df.frame_age_max_ms.max()),
                  max_interval_p95_age_ms=float(df.frame_age_p95_ms.max()),
                  cpu_target_met=bool(cpu_mean < 10 and cpu.quantile(0.95) < 10),
                  logical_cpus=manifest['logical_cpus'], post_warmup_intervals=len(df))
    result['cpu_accounting_complete'] = bool(df.cpu_accounting_complete.astype(str).str.lower().eq('true').all())
    result['profiling_enabled'] = bool(manifest.get('profiling_enabled'))
    if result['fallback'] or not result['cpu_accounting_complete'] or result['profiling_enabled']:
        result['cpu_target_met'] = False
    return manifest, df, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--plot', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    results = [summarize(path) for path in args.runs]
    report = [result for _, _, result in results]
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + '\n')
    if len(results) > 1:
        identities = [manifest['identity'] for manifest, _, _ in results]
        for field in ['fan_sha256', 'detector_state_sha256', 'schema', 'precision', 'calibration']:
            if any(identity.get(field) != identities[0].get(field) for identity in identities[1:]):
                print(f'WARNING: {field} differs; this is not an isolated CPU/CUDA comparison.')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for _, df, result in results:
            label = f"{result['backend']} / {result['profile']}"
            axes[0].plot(df.elapsed_s, df.cpu_machine_percent, label=label)
            axes[1].plot(df.elapsed_s, df.frame_age_p95_ms, label=label)
        axes[0].axhline(10, color='black', linestyle='--', label='10% target')
        axes[0].set_ylabel('Normalized process-tree CPU %')
        axes[1].set_ylabel('Per-interval p95 frame age (ms)')
        axes[1].set_xlabel('Wall time (seconds)')
        for axis in axes:
            axis.legend(); axis.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()

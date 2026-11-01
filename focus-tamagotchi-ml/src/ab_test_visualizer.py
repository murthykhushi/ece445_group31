import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Verify the telemetry logs exist before processing
    if not os.path.exists('telemetry_baseline.csv') or not os.path.exists('telemetry_optimized.csv'):
        print("Error: Missing CSV files. You need to run the A/B test first:")
        print("1. python main.py --mode baseline")
        print("2. python main.py --mode optimized")
        return

    # Load the telemetry data
    df_cpu = pd.read_csv('telemetry_baseline.csv')
    df_gpu = pd.read_csv('telemetry_optimized.csv')

    # Normalize timestamps to start at 0 seconds for clean graphing
    df_cpu['Time_Elapsed'] = df_cpu['Timestamp'] - df_cpu['Timestamp'].iloc[0]
    df_gpu['Time_Elapsed'] = df_gpu['Timestamp'] - df_gpu['Timestamp'].iloc[0]

    # Set up the matplotlib figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: CPU Proof Chart
    ax1.plot(df_cpu['Time_Elapsed'], df_cpu['CPU_Percent'], label='Baseline (CPU)', alpha=0.7, color='red')
    ax1.plot(df_gpu['Time_Elapsed'], df_gpu['CPU_Percent'], label='Optimized (Hardware Accelerated)', alpha=0.9, color='green')
    ax1.axhline(y=10, color='black', linestyle='--', label='10% Requirement Constraint')
    ax1.set_title('Host CPU Utilization Over Time')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Latency Proof Chart
    ax2.plot(df_cpu['Time_Elapsed'], df_cpu['Loop_Time_ms'], label='Baseline (CPU)', alpha=0.7, color='red')
    ax2.plot(df_gpu['Time_Elapsed'], df_gpu['Loop_Time_ms'], label='Optimized (Hardware Accelerated)', alpha=0.9, color='blue')
    ax2.set_title('Pipeline Latency (Loop Execution Time)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xlabel('Time Elapsed (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Save the graph
    plt.tight_layout()
    plt.savefig('architecture_improvements.png', dpi=300)
    print("Graph successfully saved to 'architecture_improvements.png'\n")

    # --- CALCULATE HARD NUMBERS FOR THE REPORT ---
    avg_cpu_base = df_cpu['CPU_Percent'].mean()
    avg_cpu_opt = df_gpu['CPU_Percent'].mean()
    avg_loop_base = df_cpu['Loop_Time_ms'].mean()
    avg_loop_opt = df_gpu['Loop_Time_ms'].mean()

    # Prevent division by zero if testing returns 0
    cpu_reduction = ((avg_cpu_base - avg_cpu_opt) / avg_cpu_base) * 100 if avg_cpu_base > 0 else 0
    latency_reduction = ((avg_loop_base - avg_loop_opt) / avg_loop_base) * 100 if avg_loop_base > 0 else 0

    print("--- EFFICIENCY METRICS FOR REPORT ---")
    print(f"Baseline CPU:  {avg_cpu_base:.2f}%")
    print(f"Optimized CPU: {avg_cpu_opt:.2f}%")
    print(f"CPU Reduction: {cpu_reduction:.1f}%")
    print(f"Target Constraint Met: {'Yes' if avg_cpu_opt < 10.0 else 'No'} (<10% required)\n")
    
    print(f"Baseline Latency:  {avg_loop_base:.1f}ms")
    print(f"Optimized Latency: {avg_loop_opt:.1f}ms")
    print(f"Latency Reduction: {latency_reduction:.1f}%")

if __name__ == "__main__":
    main()
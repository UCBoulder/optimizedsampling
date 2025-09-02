import subprocess

def get_free_cpus(threshold=75):
    """Return number of CPUs with CPU usage below threshold percent."""
    print("Getting number of free CPUs...")
    result = subprocess.run(
        ["mpstat", "-P", "ALL", "1", "1"],
        capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    free_cpus = 0
    for line in lines:
        parts = line.split()
        if len(parts) > 2:  # skip headers, 'all' line
            try:
                idle = float(parts[-1])  # %idle column
                usage = 100.0 - idle
                if usage < threshold:
                    free_cpus += 1
            except ValueError:
                continue
    print(f"Num free cpus: {free_cpus}")
    return max(free_cpus, 1)  # at least use 1 cpu

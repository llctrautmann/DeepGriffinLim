import subprocess
import re
import time
import torch
from hyperparameter import hp

# List of loss types
loss_types = ['L1', 'phase', 'gdl', 'ifr', 'all_l1']

num_gpus = torch.cuda.device_count()

def is_gpu_available(gpu_id, memory_limit=35000, verbose=False):
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        free_memory = int(re.search(r"\d+", result.stdout).group())
        if verbose:
            print(f"Checking GPU {gpu_id}:")
            print(f" - Free memory: {free_memory} MiB")
            print(f" - Memory limit: {memory_limit} MiB")
        is_available = free_memory > memory_limit
        print(f" - Is available: {is_available}\n")
        return is_available
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"An error occurred while checking the availability of GPU {gpu_id}: {str(e)}")
        return False

def run_script(gpu_id, device, loss_type):
    script_name = "main.py"
    cmd = f"python3 {script_name} --device {device} --loss_type {loss_type}"
    try:
        # Use Popen instead of run
        process = subprocess.Popen(cmd, shell=True)
        return process  # Return the process
    except Exception as e:
        print(f"An error occurred while running the script: {str(e)}")
        return None

if __name__ == '__main__':
    processes = []  # List to store active processes

    while loss_types:
        for gpu_id in range(num_gpus):
            if len(loss_types) == 0:
                break
            if is_gpu_available(gpu_id):
                loss_type = loss_types.pop(0)

                # Create Cuda tag for the device
                device_tag = f'cuda:{str(gpu_id)}'

                # Add code to run the main script
                process = run_script(gpu_id, device_tag, loss_type)
                if process:
                    processes.append(process)

                # Wait for the process to finish if too many processes are active
                while len(processes) >= num_gpus:
                    for p in processes:
                        if p.poll() is not None:  # Process has finished
                            processes.remove(p)
                    time.sleep(10)  # Sleep for a bit before checking again

            else:
                time.sleep(10)
            time.sleep(60)

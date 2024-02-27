import subprocess
import re
import time

def check_gpu_availability():
    """
    Check GPU availability by parsing the output of gpustat command.
    Returns a list of available GPU IDs.
    """
    gpu_info = subprocess.check_output(['gpustat']).decode('utf-8')
    gpu_lines = gpu_info.split('\n')

    available_gpus = []
    for i, line in enumerate(gpu_lines):
        if 'GeForce' in line:  # Adjust this condition based on your GPU model name
            utilization = int(re.findall(r'\d+ %', line)[0][:-1])
            memory_usage = int(re.findall(r'\d+ / \d+ MB', line)[0].split('/')[0].strip())
            if utilization < 20 and memory_usage < 3000:  # Adjust these thresholds as needed
                available_gpus.append(i)
    return available_gpus

def start_training_script(gpu_ids):
    """
    Start the training script with the available GPU IDs.
    """
    if gpu_ids:
        gpu_args = ' '.join(['-g {}'.format(gpu_id) for gpu_id in gpu_ids])
        command = 'screen -S training_session -m bash -c "train_imagenet_all.py {} --algorithms threshold"'.format(gpu_args)
        subprocess.call(command, shell=True)
    else:
        print("No available GPU found. Retrying in 30 seconds...")
        time.sleep(30)
        start_training_script(check_gpu_availability())

if __name__ == "__main__":
    start_training_script(check_gpu_availability())

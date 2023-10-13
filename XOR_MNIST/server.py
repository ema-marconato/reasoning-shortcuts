import submitit
from main import parse_args, main
from example.xor_main import prepare_args
from experiments import *
import os, sys
from example.xor_main import xor_run

conf_path = os.getcwd() + "."
sys.path.append(conf_path)

if __name__ == '__main__':
    #start_main()
    args = parse_args() # 
#     args = prepare_args() #  parse_args() # 
    executor = submitit.AutoExecutor(folder="./logs", slurm_max_num_timeout=30)
    executor.update_parameters(
            mem_gb=4,
            gpus_per_task=1,
            tasks_per_node=1,  # one task per GPU
            cpus_per_gpu=4,
            nodes=1,
            timeout_min=20,
            # Below are cluster dependent parameters
            slurm_partition="yours",
            slurm_signal_delay_s=120,
            slurm_array_parallelism=4)

    experiments=launch_short_search(args) 
    executor.update_parameters(name="mnist")
    jobs = executor.map_array(main,experiments)

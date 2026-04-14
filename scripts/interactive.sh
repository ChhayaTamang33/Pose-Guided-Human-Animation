#!/bin/bash

srun -K -p RTXA6000 --nodes=1 --gpus=1 --ntasks=1 --mem=64G --immediate=300 --cpus-per-task=4  --time=00-01:00:00 --job-name="chhaya"  \
        --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER \
        --container-image=/netscratch/ctamang/enroot/moore_train.sqsh --container-workdir="`pwd`"   \
        --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" --pty /bin/bash  


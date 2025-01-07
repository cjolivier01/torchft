#!/bin/bash
TORCHFT_MANAGER_PORT=29512 \
  TORCHFT_LIGHTHOUSE="http://$(hostname):29510" \
  srun -N 1 --tasks-per-node=1 --reservation=colivier-dpu -p ap-dpu ./slurm_worker.sh $@

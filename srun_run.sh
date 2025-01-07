#!/bin/bash

MAX_RESTARTS=3

# Function to parse SLURM_NODELIST and extract the first node
get_first_slurm_node() {
    local nodelist="$SLURM_NODELIST"
    local expanded_nodes

    # Check if SLURM_NODELIST is set
    if [[ -z "$nodelist" ]]; then
        echo "SLURM_NODELIST environment variable is not set."
        return 1
    fi

    # Use scontrol to expand the node list and capture output
    expanded_nodes=$(scontrol show hostnames "$nodelist")
    
    # Get the first line from the expanded list
    first_node=$(echo "$expanded_nodes" | head -n 1)

    echo "$first_node"
}

# /bin/env | grep SLURM_ | sort
torchrun \
  --nodes=${SLURM_NNODES} \
  --nproc-per-node=${SLURM_NTASKS_PER_NODE} \
  --max-restarts=${MAX_RESTARTS} \
  --rdzv-id=${SLURM_JOBID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$(get_first_slurm_node)" \
  $@

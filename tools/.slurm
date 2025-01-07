#!/bin/bash
if [ "${CONDA_DEFAULT_ENV}" != "ubuntu" ]; then
  source ~/src/ai/tools/.slurm
  mamba activate ubuntu
fi

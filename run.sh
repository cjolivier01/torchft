TORCHFT_MANAGER_PORT=29512 TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port 29501 --nnodes 1 --nproc_per_node 1 train_ddp.py

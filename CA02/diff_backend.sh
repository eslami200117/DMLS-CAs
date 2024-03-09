torchrun --standalone --nnodes=1 --nproc_per_node=2 different_backend.py 32 nccl >> diff_backend.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 different_backend.py 128 nccl >> diff_backend.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 different_backend.py 32 gloo >> diff_backend.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 different_backend.py 128 gloo >> diff_backend.out

torchrun --standalone --nnodes=1 --nproc_per_node=2 batch_size.py 16 >> batch_size.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 batch_size.py 32 >> batch_size.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 batch_size.py 64 >> batch_size.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 batch_size.py 128 >> batch_size.out
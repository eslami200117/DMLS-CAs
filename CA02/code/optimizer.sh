torchrun --standalone --nnodes=1 --nproc_per_node=2 optimizer.py Adam >> optimizer.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 optimizer.py SGD >> optimizer.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 optimizer.py Adagrad >> optimizer.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 optimizer.py RMSprop >> optimizer.out

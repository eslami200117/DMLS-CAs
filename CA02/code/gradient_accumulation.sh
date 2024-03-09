torchrun --standalone --nnodes=1 --nproc_per_node=2 classifier_torchrun_acc.py 16 >> gradient_accumulation.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 classifier_torchrun_acc.py 32 >> gradient_accumulation.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 classifier_torchrun_acc.py 64 >> gradient_accumulation.out
torchrun --standalone --nnodes=1 --nproc_per_node=2 classifier_torchrun_acc.py 128 >> gradient_accumulation.out
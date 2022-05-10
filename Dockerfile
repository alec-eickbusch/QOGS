FROM nvcr.io/nvidia/tensorflow:22.03-tf2-py3

RUN pip install pylint qutip tensorboard

ENV TF_XLA_FLAGS='--tf_xla_auto_jit=2'
ENV TF_GPU_ALLOCATOR='cuda_malloc_async'
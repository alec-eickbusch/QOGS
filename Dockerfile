FROM nvcr.io/nvidia/tensorflow:22.06-tf2-py3

RUN pip install --upgrade pip
RUN pip install pylint qutip tensorboard tensorboard-plugin-profile
# RUN pip install workspaces/QOGS/

ENV TF_XLA_FLAGS='--tf_xla_auto_jit=2'
ENV TF_GPU_ALLOCATOR='cuda_malloc_async'
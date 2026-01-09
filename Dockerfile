FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install required packages.
RUN apt-get update && apt-get install -y \
	gcc \
	git \
	libtinfo5 \
	wget \
	cmake \
	htop \
	unzip \
	vim \
    nano \
    gdb

# Install NVHPC
RUN echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list && apt update && apt install -y nvhpc-23-5 environment-modules

WORKDIR /usr/src/fhe/
ENV PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda/12.1/bin${PATH:+:${PATH}}
ENV PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/compilers/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
COPY . .
ENV CPATH /usr/local/include:$CPATH

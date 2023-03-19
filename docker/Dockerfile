# USE build.sh FOR FAST AND CONVENIENT DOCKER BUILD

FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu16.04

# apt tools
RUN APT_INSTALL="apt install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' \
        /etc/apt/sources.list && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        zip \
        unzip \
        vim \
        nano \
        wget \
        curl \
        git \
        aria2 \
        apt-transport-https \
        openssh-client \
        openssh-server \
        libopencv-dev \
        libsnappy-dev \
        libopenblas-dev \
        libatlas-base-dev \
        tzdata \
        iputils-ping \
        net-tools \
        htop


# install miniconda
RUN curl -o ~/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/anaconda.sh && \
    bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# change sources and install python
RUN conda config --set show_channel_urls yes && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda install -y python=3.7 && \
    conda update --all

# add ssh key for git
ARG SSH_KEY

RUN mkdir -p /root/.ssh/ && \
    echo "$SSH_KEY" > /root/.ssh/id_rsa && \
    chmod -R 600 /root/.ssh && \
    ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

# install pytorch
RUN pip install torch \
                --extra-index-url https://download.pytorch.org/whl/cu116

# install utils
RUN pip install tianshou==0.4.10 \
                gym==0.21.0 \
                pyglet==1.5.27 \
                matplotlib \
                tensorboard-plugin-customizable-plots

# finally, remove ssh files
RUN rm -r /root/.ssh

WORKDIR /root
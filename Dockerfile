FROM nvidia/cuda:12.8-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml \
    && conda clean -afy

ENV CONDA_DEFAULT_ENV=myenv
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

COPY . .

CMD ["python", "-c", "print('Container ready!')"]
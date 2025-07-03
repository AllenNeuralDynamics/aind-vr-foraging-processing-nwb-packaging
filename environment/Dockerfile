# hash:sha256:862591ebb9d26ceba468af9aa1f0a5ef5be8954e57ac1e2730c59ad7459036f0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
        libglib2.0-0=2.72.4-0ubuntu2.5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.11.1 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.3.0 \
    scipy==1.13.1 \
    -e git+https://github.com/rly/ndx-events.git@0d4e8efdcf1ab2eb9da9ca078b4522fcb82fa8a6#egg=ndx-events

COPY postInstall /
RUN /postInstall

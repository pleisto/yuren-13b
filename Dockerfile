# This Dockerfile is  work in progress
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive


RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt/archives \
    apt update && \
    apt upgrade -y && \
    apt install --fix-missing -y git curl dos2unix \
        libcudnn8 libcupt-common cuda-cupti-12-2


ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash

WORKDIR /src

SHELL [ "bash", "-c" ]

RUN ${RYE_HOME}/self/bin/pip install -U pip==23.1

COPY . .

RUN rye sync


FROM base

WORKDIR /src
# put the model file in here
VOLUME [ "/pleisto/yuren-13b" ]
ENV YUREN_WEB_TITLE "羽人-13b"
# Expose for web service
EXPOSE 7860

RUN ln -s \
    /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcupti.so.2023.2.0 \
    /usr/lib/libcupti.so.12.2

ENTRYPOINT [ "rye","run" ,"webui" ,"/pleisto/yuren-13b"]

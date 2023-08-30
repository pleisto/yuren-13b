# This Dockerfile is  work in progress
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive


RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt/archives \
    apt update && \
    apt upgrade -y && \
    apt install --fix-missing -y git curl dos2unix \
        libcudnn8 libcupt-common cuda-cupti-11-7


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

ENTRYPOINT [ "rye","run" ,"webui" ,"/pleisto/yuren-13b"]
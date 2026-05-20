ARG base_image=tensorflow/tensorflow:2.19.0
ARG install_extras=

FROM ${base_image}

LABEL maintainer="Wout Decrop (VLIZ)"
LABEL org.opencontainers.image.title="planktonclass inference"
LABEL org.opencontainers.image.description="Inference image for a packaged planktonclass model run"

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DISABLE_AUTHENTICATION_AND_ASSUME_AUTHENTICATED_USER=yes \
    DEEPAAS_V2_MODEL=planktonclass \
    planktonclass_CONFIG=/srv/project/config.yaml

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/app

COPY pyproject.toml requirements.txt README.md VERSION MANIFEST.in /tmp/app/
COPY planktonclass /tmp/app/planktonclass

RUN python3 --version && \
    pip3 install --no-cache-dir --upgrade pip "setuptools<60.0.0" wheel && \
    pip3 install --no-cache-dir "keras==3.14.0" && \
    pip3 install --no-cache-dir --ignore-installed blinker blinker && \
    if [ -n "${install_extras}" ]; then \
        pip3 install --no-cache-dir "/tmp/app[${install_extras}]"; \
    else \
        pip3 install --no-cache-dir /tmp/app; \
    fi

WORKDIR /srv/project

COPY project/config.yaml /srv/project/config.yaml
COPY project/models /srv/project/models

RUN mkdir -p /tmp/planktonclass-predictions

EXPOSE 5000

CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]

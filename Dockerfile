# Reproducible Python 3.11 environment for the fridge detector pipeline.
#
# This is a TOOLING image you run locally or on a GPU box. It is not a web service
# and must not be deployed to Render: the model runs on the phone, and this repo's
# only output is exports/fridge-detector-vN.tflite + fridge-labels-vN.json.
#
# Targets:
#   test   core tools + pytest            docker build --target test  -t fridge-vision:test .
#   data   + FiftyOne / Roboflow           docker build --target data  -t fridge-vision:data .
#   train  + MediaPipe Model Maker / TF    docker build --target train -t fridge-vision:train .
#          GPU (NVIDIA + WSL2):            docker build --target train --build-arg GPU=true -t fridge-vision:train-gpu .
#
# Run with the repo mounted so data/, runs/ and exports/ stay on your disk:
#   docker run --rm -it -v "${PWD}:/workspace" fridge-vision:train python scripts/train.py --data data/processed/v1

FROM python:3.11-slim AS base

# OpenCV + MediaPipe native deps (same set as the form-coach service)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgles2 libegl1 libgbm1 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# ------------------------------------------------------------------ test
FROM base AS test
COPY . .
CMD ["python", "-m", "pytest"]

# ------------------------------------------------------------------ data
FROM base AS data
COPY requirements-data.txt ./
RUN pip install -r requirements-data.txt
COPY . .
CMD ["python", "scripts/download_open_images.py", "--help"]

# ------------------------------------------------------------------ train
FROM base AS train
ARG GPU=false
COPY requirements-train.lock ./
RUN pip install -r requirements-train.lock \
    && if [ "$GPU" = "true" ]; then pip install "tensorflow[and-cuda]==2.15.1"; fi
COPY . .
# Fail the build early if the ML stack doesn't import.
RUN python -c "import mediapipe, tensorflow as tf; from mediapipe_model_maker import object_detector; print('tf', tf.__version__, 'mediapipe', mediapipe.__version__)"
CMD ["python", "scripts/train.py", "--help"]

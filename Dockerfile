FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates ffmpeg libgl1 libglib2.0-0 \
    && python3 -m venv /opt/venv && rm -rf /var/lib/apt/lists/*
WORKDIR /iabot
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY comfy-source.json install_comfy.py ./
RUN python install_comfy.py \
    && pip install --no-cache-dir -r /comfyui/requirements.txt 'transformers>=4.50.3,<5' 'huggingface-hub<1.0' runpod==1.12.0 requests==2.34.2
COPY worker.py validation.py start-iabot.sh test_validation.py ./
COPY templates/03_Prueba_GPU_corta_api.json ./test-workflow.json
RUN python -m unittest -q test_validation
COPY models.json download_models.py ./
RUN python download_models.py \
    && cd /comfyui && python main.py --quick-test-for-ci --cpu --disable-all-custom-nodes
RUN chmod 755 /iabot/start-iabot.sh
CMD ["bash", "/iabot/start-iabot.sh"]

FROM paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git \
    libgl1 libglib2.0-0 \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel \
 && python3 -m pip install --no-cache-dir "paddleocr<3.0" opencv-python \
    fastapi uvicorn[standard] python-multipart \
    pdf2image pillow

COPY api.py /work/api.py

EXPOSE 8000
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

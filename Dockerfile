FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# 필수 패키지 및 Python 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 설치
ENV TORCH_INSTALL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
RUN pip3 install --no-cache-dir $TORCH_INSTALL

# Triton 클라이언트 다운로드 및 설치
# COPY tritonserver2.43.0-igpu.tar /tmp/
# RUN tar -xzvf /tmp/tritonserver2.43.0-igpu.tar -C /opt/ && ls /opt/
# RUN python3 -m pip install --upgrade /opt/tritonserver/clients/python/tritonclient-2.43.0-py3-none-manylinux2014_aarch64.whl[all]

# Flash Attention 설치
ENV FLASH="http://jetson.webredirect.org/jp6/cu122/+f/b94/48fc2bf0a7532/flash_attn-2.5.7-cp310-cp310-linux_aarch64.whl"
RUN pip3 install --no-cache-dir $FLASH

# 추가 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    libopenblas-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

# requirements.txt에서 패키지 설치
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 모든 파일 복사
COPY . .


CMD ["python3", "script_base.py"]
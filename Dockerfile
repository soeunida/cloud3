FROM python:3.10-slim
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0


#WORKDIR /home/hayoung/cloud

COPY requirements.txt ./

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until
RUN apt-get update &&  apt-get install -y git
RUN set -xe && apt-get -yqq update && apt-get -yqq install python3-pip && pip3 install --upgrade pip


#ENV TORCH_INSTALL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+40ec155e58.nv24.03.13384722-cp310-cp310-linux_aarch64.whl"
ENV TORCH_INSTALL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl"
RUN pip3 install --no-cache $TORCH_INSTALL
RUN apt-get install -y --no-install-recommends libopenmpi-dev libopenblas-base
#ENV FLASH="https://static.abacus.ai/pypi/abacusai/gh200-llm/flash_attn-2.3.6-cp310-cp310-linux_aarch64.whl"
ENV FLASH="http://jetson.webredirect.org/jp6/cu122/+f/b94/48fc2bf0a7532/flash_attn-2.5.7-cp310-cp310-linux_aarch64.whl"
RUN pip3 install --no-cache $FLASH

RUN pip install --no-cache -r requirements.txt

RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn

ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=/usr/local/cuda-12.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

COPY . .

CMD  ["python3", "script_server.py"]

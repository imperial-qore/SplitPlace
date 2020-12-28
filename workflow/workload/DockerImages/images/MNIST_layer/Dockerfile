FROM ubuntu:18.04

ADD main.py /

WORKDIR /root

RUN \
    apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    libtool \
    time \
    bc \
    python3 \
    python3-pip \
    wget

RUN \
    pip3 install torch torchvision 
    
RUN \
    cd Aeneas/aeneas && \
    bash install_dependencies.sh && \
    pip install -r requirements.txt && \
    python setup.py build_ext --inplace
    
COPY assets assets

CMD [ "python", "./main.py" ]

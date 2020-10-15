from nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/build/darknet:${PATH}"

# Install needed packages
RUN apt-get update && \
    apt-get install -y git wget python3-opencv libopencv-dev

# Create build base dir
RUN mkdir build

# Compile and install cmake -- needed to compile yolov4 project
RUN cd /build && \
    wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2.tar.gz && \
    tar xzvf cmake-3.18.2.tar.gz && \
    cd cmake-3.18.2 && \
    ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF && \
    make && \
    make install

# RUN git clone https://github.com/roboflow-ai/darknet.git # Varios improvements no codigo do AlexeyAB/darknet
# Download Yolo project
RUN cd /build && \
    git clone https://github.com/AlexeyAB/darknet.git

# OBS.: Verificar no arquivo Makefile se as instrucoes do ARCH ativas
#       sao compativeis com o modelo da GPU utilizada.

# Compile Darknet Yolo-v4
RUN cd /build/darknet && \
    sed -i -e 's/GPU=0$/GPU=1/' Makefile && \
    sed -i -e 's/CUDNN=0$/CUDNN=1/' Makefile && \
    sed -i -e 's/OPENCV=0$/OPENCV=1/' Makefile && \
    sed -i -e 's/AVX=0$/AVX=1/' Makefile && \
    sed -i -e 's/OPENMP=0$/OPENMP=1/' Makefile && \
    sed -i -e 's/LIBSO=0$/LIBSO=1/' Makefile && \
    sed -i -e 's/\(^cmake_minimum_required.*\)/\1\ninclude_directories(\/usr\/local\/cuda-10.2\/compat)/' CMakeLists.txt && \
    make

CMD ["bash"]

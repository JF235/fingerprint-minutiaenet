# Use an older Ubuntu version compatible with Python 2.7 and ancient TensorFlow
FROM ubuntu:16.04

# Update apt and install basic system requirements and Python 2.7 packages natively.
# Ubuntu 16.04 natively ships with Python 2.7, and installing packages via apt
# bypasses the extremely slow conda resolution for deprecated Python versions.
RUN apt-get update && apt-get install -y \
    python2.7 \
    python-pip \
    python-numpy \
    python-scipy \
    python-matplotlib \
    python-opencv \
    python-pydot \
    python-h5py \
    graphviz \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to a version that supports Py27
RUN pip install --no-cache-dir --upgrade "pip<21.0"

# Install TensorFlow 1.7.0 and Keras 2.1.6
# h5py is already installed via apt (python-h5py) which is safe and compatible.
RUN pip install --no-cache-dir \
    tensorflow==1.7.0 \
    Keras==2.1.6 \
    scikit-image==0.14.2

WORKDIR /workspace

CMD ["/bin/bash"]

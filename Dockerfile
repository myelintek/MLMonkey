FROM tensorflow/tensorflow:1.13.1-gpu-py3

WORKDIR /workspace
ENV HOME /workspace
#ARG tensorflow_pip_spec="tf-nightly-gpu"

# Add google-cloud-sdk to the source list
RUN apt-get install -y curl
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update

# Install extras needed by most models
RUN apt-get install -y \
      git \
      build-essential \
      software-properties-common \
      ca-certificates \
      wget \
      htop \
      zip \
      google-cloud-sdk \
      vim \
      unzip \
      time \
      lighttpd \
      lshw \
      bc \
      pv \
      libnccl2=2.4.7-1+cuda10.0 \
      libnccl-dev=2.4.7-1+cuda10.0 \
      libcudnn7-dev=7.4.1.5-1+cuda10.0 \
      cuda-curand-10-0 \
      cuda-curand-dev-10-0


# Install / update Python and Python3
#RUN apt-get install -y --no-install-recommends \
#      python3 \
#      python3-dev \
#      python3-pip \
#      python3-setuptools \
#      python3-venv


# Setup Python3 environment
#RUN pip3 install --upgrade pip==9.0.1
# setuptools upgraded to fix install requirements from model garden.
#RUN pip3 install --upgrade setuptools google-api-python-client google-cloud google-cloud-bigquery
#RUN pip3 install wheel absl-py
#RUN pip3 install --upgrade --force-reinstall ${tensorflow_pip_spec}

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 freeze

RUN wget -O /tmp/docker.tgz https://download.docker.com/linux/static/stable/x86_64/docker-18.09.4.tgz ; \
    tar zxvf /tmp/docker.tgz -C /tmp/ ; \
    cp /tmp/docker/docker /usr/bin/docker ; \
    rm -rf /tmp/docker*

RUN cd / ; mkdir run_benchmarks ; cd run_benchmarks ; \ 
git clone -b cnn_tf_v1.13_compatible https://github.com/tensorflow/benchmarks.git ; \
git clone https://github.com/mlperf/training.git ; \
git clone https://github.com/myelintek/results.git

COPY scripts /scripts
RUN mkdir -p /web && \
    cd /scripts && \
    mv index.html /web && \
    mv lighttpd.conf /etc/lighttpd/lighttpd.conf 

ENTRYPOINT bash /scripts/run.sh

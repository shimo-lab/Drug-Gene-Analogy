FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
  
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    vim \
    wget

# Copy files from host to the image.
COPY requirements.txt /tmp/requirements.txt

# Install python package, remove copied file and cache.
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Language settings
ENV LANG=C.UTF-8
ENV LANGUAGE=en_US

# Create the user.
ARG USERNAME
ARG USER_UID
ARG USER_GID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/$USERNAME

WORKDIR /workspace
RUN chmod 777 /workspace

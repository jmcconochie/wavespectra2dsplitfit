# Base image
FROM python:3.10-slim
LABEL "Name"="wavespectra2dsplitfit:deploy"

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip
# For development
RUN apt-get install -y vim nodejs

# Software add ons
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# User
RUN adduser shell --disabled-login
RUN echo "shell:wavespec" | chpasswd

# Sudo perms
RUN apt-get -y install sudo
RUN usermod -aG sudo shell

# Ports and run command
USER shell
WORKDIR /home/shell/
COPY wavespectra2dsplitfit .

#CMD ["bash"]

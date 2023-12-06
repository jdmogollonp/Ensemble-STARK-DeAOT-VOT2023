# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app


# Upgrade pip
RUN pip install --upgrade pip


# Run installation scripts for Stark and MixFormer
RUN cd /usr/src/app/Stark && bash install.sh install_pytorch17.sh
RUN cd /usr/src/app/MixFormer && bash install_pytorch17.sh

# Install segment-anything and vot-toolkit
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install vot-toolkit

# Install libgl1-mesa-glx for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx


RUN touch /usr/src/app/container_alive.txt
CMD tail -f /usr/src/app/container_alive.txt



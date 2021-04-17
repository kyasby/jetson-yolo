FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

RUN apt update \
&&	apt install -y \
	sudo \
	vim \
	xterm

RUN pip3 install -U pip

RUN git clone https://github.com/kyasby/jetson-yolo.git requirements

WORKDIR requirements

RUN pip3 install -r requirements.txt

FROM python:3.8-slim-buster

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /src

COPY requirements.txt .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip --upgrade pip setuptools wheel && pip install numpy==1.21.3
RUN pip install -r requirements.txt

COPY src/ .

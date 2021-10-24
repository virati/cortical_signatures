FROM python:3.8.9

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt

copy src/ .

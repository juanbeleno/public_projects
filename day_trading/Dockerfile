# syntax=docker/dockerfile:1
FROM python:3.10.2-slim
LABEL org.opencontainers.image.authors="@juanbeleno"

# libgomp1 is a program required by LightGBM
RUN apt update && \
    apt-get install libgomp1 && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ADD /day_trading /app/day_trading
ADD setup.py /app
ADD requirements.txt /app

# Update pip
RUN pip install --upgrade pip

# WORKAROUND: There are some problems when installing the requirements from the setup.py
RUN pip install -r requirements.txt

RUN python setup.py install

ENTRYPOINT ["day_trading",  "train-models"]
# ENTRYPOINT ["python",  "train.py"]
# CMD day_trading train-models

FROM python:3.9-slim

RUN mkdir /code
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
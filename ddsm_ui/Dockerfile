FROM python:3.7-slim
RUN apt-get update -y
RUN apt-get install -y libsm6 libxrender-dev libglib2.0-0 libxext-dev
COPY requirements.txt /tmp/
COPY ui /ui
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt
WORKDIR /ui
ENTRYPOINT [ "python", "./app.py" ]

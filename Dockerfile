# Dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    openjdk-11-jdk \
    maven \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Setup working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# Install Defects4J
RUN git clone https://github.com/rjust/defects4j.git /opt/defects4j
ENV PATH="/opt/defects4j/framework/bin:${PATH}"
RUN cd /opt/defects4j && ./init.sh

# Copy source code
COPY src/ /app/src/
COPY lib/ /app/lib/
COPY main.py /app/
COPY server.py /app/
COPY run_overfitting_detection.py /app/
COPY configs/ /app/configs/
COPY data/perfect_fl.json /app/data/

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python3", "main.py"]
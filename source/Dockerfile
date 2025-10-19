# Dockerfile
FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    openjdk-11-jdk \
    maven \
    git \
    wget \
    curl

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Install Defects4J
RUN git clone https://github.com/rjust/defects4j.git /opt/defects4j
ENV PATH="/opt/defects4j/framework/bin:${PATH}"
RUN cd /opt/defects4j && ./init.sh

# Install Node.js for JavaScript support
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

# Setup working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/
COPY main.py /app/
COPY configs/ /app/configs/

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python3", "main.py"]
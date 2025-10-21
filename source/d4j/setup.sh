#!/bin/bash
# setup.sh

echo "Setting up Automated Bug Fixing System..."

# Create directory structure
mkdir -p data/{defects4j,bugsinpy,vector_db}
mkdir -p results
mkdir -p configs
mkdir -p tests
mkdir -p lib

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download and setup Defects4J
echo "Setting up Defects4J..."
if [ ! -d "defects4j" ]; then
    git clone https://github.com/rjust/defects4j.git
    cd defects4j
    ./init.sh
    cd ..
fi

# Download CodeBERT model
echo "Downloading CodeBERT model..."
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('microsoft/codebert-base')
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
print('CodeBERT model downloaded successfully')
"

# Setup BugsInPy
echo "Setting up BugsInPy..."
if [ ! -d "bugsinpy" ]; then
    git clone https://github.com/soarsmu/BugsInPy.git bugsinpy
fi

# Download sample projects for vector DB
echo "Downloading sample projects..."
cd data/defects4j
for project in Chart Closure Lang Math Mockito Time; do
    echo "Downloading $project..."
    defects4j checkout -p $project -v 1f -w ./$project
done
cd ../..

# Build initial vector databases
echo "Building vector databases (this may take a while)..."
python3 main.py --mode preprocess --benchmark defects4j

echo "Setup complete!"
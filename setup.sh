#!/bin/bash
# setup.sh

set -e

echo "Setting up Automated Bug Fixing System..."

# Create directory structure
mkdir -p data/{defects4j,vector_db}
mkdir -p results
mkdir -p configs
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

# Add Defects4J to PATH
export PATH="$PWD/defects4j/framework/bin:$PATH"
echo ""
echo "Add the following to your shell profile (~/.bashrc or ~/.zshrc):"
echo "  export PATH=\"$(pwd)/defects4j/framework/bin:\$PATH\""
echo ""

# Verify Defects4J installation
if command -v defects4j &> /dev/null; then
    echo "Defects4J installed successfully."
else
    echo "WARNING: Defects4J not found in PATH. Please add it manually."
fi

# Download CodeBERT model
echo "Downloading CodeBERT model..."
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('microsoft/codebert-base')
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
print('CodeBERT model downloaded successfully')
"

# Checkout sample Defects4J projects for vector DB building
echo "Checking out sample Defects4J projects..."
cd data/defects4j
for project in Chart Closure Lang Math Mockito Time; do
    if [ ! -d "./$project" ]; then
        echo "Checking out $project..."
        defects4j checkout -p $project -v 1b -w ./$project
    fi
done
cd ../..

# Build initial vector databases
echo "Building vector databases (this may take a while)..."
python3 main.py --mode preprocess --benchmark defects4j

echo ""
echo "Setup complete!"
echo "Don't forget to set your OpenAI API key:"
echo "  export OPENAI_API_KEY='your-api-key'"
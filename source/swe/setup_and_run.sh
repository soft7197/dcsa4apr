#!/bin/bash
# setup_and_run.sh - Complete setup and execution script

set -e

echo "=========================================="
echo "Agent-Based APR System Setup"
echo "=========================================="

# 1. Create directory structure
echo "Creating directory structure..."
mkdir -p workspace
mkdir -p results
mkdir -p faiss_index
mkdir -p datasets

# 2. Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 3. Download SWE-bench Lite dataset
echo "Downloading SWE-bench Lite dataset..."
cd datasets

# Download the lite dataset
if [ ! -f "swebench_lite.json" ]; then
    echo "Fetching SWE-bench Lite from Hugging Face..."
    python3 << 'EOF'
from datasets import load_dataset
import json

# Load SWE-bench Lite
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Convert to JSON
data = []
for item in dataset:
    data.append({
        "instance_id": item["instance_id"],
        "repo": item["repo"],
        "base_commit": item["base_commit"],
        "problem_statement": item["problem_statement"],
        "hints_text": item.get("hints_text", ""),
        "test_patch": item["test_patch"],
        "patch": item["patch"],
        "version": item["version"],
        "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": item.get("PASS_TO_PASS", [])
    })

with open("swebench_lite.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Downloaded {len(data)} instances")
EOF
fi

cd ..

# 4. Set OpenAI API Key
echo ""
echo "=========================================="
echo "API Key Configuration"
echo "=========================================="
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    echo "Or add it to .env file"
    read -p "Enter your OpenAI API key now (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "API key set for this session"
    fi
fi

# 5. Create .env file template
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
OPENAI_API_KEY=your-api-key-here
EOF
    echo "Created .env template file"
fi

# 6. Run the system
echo ""
echo "=========================================="
echo "Starting APR System"
echo "=========================================="
python main.py

echo ""
echo "=========================================="
echo "Setup and execution complete!"
echo "Check results/ directory for outputs"
echo "=========================================="
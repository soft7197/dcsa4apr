Deployment and Usage Instructions
Installation
bash# Clone repository
git clone https://github.com/soft7197/dcsa4apr.git
cd dcsa4apr

# Run setup script
chmod +x setup.sh
./setup.sh

# Set API keys
export OPENAI_API_KEY="your-api-key"
Usage Examples
bash# Preprocess benchmark
python main.py --mode preprocess --benchmark defects4j

# Fix a single bug
python main.py --mode repair --benchmark defects4j --bug-id Chart-1

# Run full evaluation
python main.py --mode evaluate --benchmark defects4j --workers 8

# Compare with other tools
python main.py --mode compare --benchmark defects4j
Docker Usage
bash# Build Docker image
docker build -t bug-fixer .

# Run in Docker
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           -e OPENAI_API_KEY=$OPENAI_API_KEY \
           bug-fixer --mode repair --bug-id Chart-1
This complete implementation provides:

Preprocessing: Vector DB creation with CodeBERT embeddings
Fault Localization: Support for both automatic and perfect FL
Multi-Agent System: Context updater, generator, and hypothesis management
Dynamic Context Management: Token-aware context updates
Patch Execution: Safe execution in Docker containers
Evaluation Framework: Parallel benchmark execution
Monitoring: Performance and error tracking
Testing: Comprehensive unit tests
Configuration: Flexible YAML-based configuration
Deployment: Docker support and setup scripts

The system is modular, scalable, and ready for experimentation with different configurations and benchmarks.

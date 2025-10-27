# Dynamic Context and Static Analysis for Agent-Based Automated Program Repair

This repository contains the implementation and experimental results for our paper "Utilizing Dynamic Context and Static Analysis for Agent-Based Automated Program Repair" accepted at SAC'26.

## Overview

This work presents an agent-based automated program repair approach that orchestrates specialized LLM agents with dynamic context management to iteratively refine and generate patches. The system addresses key limitations in current LLM-based APR approaches through:

- **Agent-based architecture** with Context Updater, Generator, and Overfitting Detector agents
- **Dynamic context pool** that accumulates information from static analysis tools during repair
- **Integrated tool suite** providing diverse information sources (coverage analysis, semantic search, code extraction, call graphs, field dependencies, API usage patterns)
- **Component-level repair** with intelligent mapping between buggy methods and failing tests

## Experimental Results

### Defects4J Benchmark
- **357 total bugs fixed** (183 on v1.2, 174 on v2.0)
- **48 multi-function bugs** successfully repaired (up to 10 functions)
- 7.5% improvement over previous state-of-the-art (SRepair)

### SWE-Bench Lite
- **87 bugs fixed** across 10 Python repositories (29.9% success rate)
- **10 multi-function bugs** successfully repaired
- Demonstrates effective cross-language generalization

## Repository Structure

```
dcsa4apr/
├── src/           # Source code implementation
├── results/       # Experimental results and data
├── figures/       # Figures and visualizations from the paper
└── README.md
```

## Key Features

### Multi-Agent Architecture
- **Context Updater Agent**: Dynamically retrieves relevant information based on repair feedback
- **Generator Agent**: Produces diverse patch hypotheses informed by accumulated context
- **Overfitting Detector Agent**: Analyzes patches to prevent overly specific solutions

### Static Analysis Tools
1. Coverage Runner - Test execution with coverage instrumentation
2. Similar Method Search - Semantic code search using vector database
3. Code Extractor - Multi-granularity code retrieval
4. Call Graph Builder - Method relationship analysis
5. Field Dependency Analyzer - Data flow tracking
6. API Usage Finder - Usage pattern mining

### Context Management
- **Static Context**: Original buggy code, failing tests, bug reports
- **Dynamic Context**: Tool outputs, tried hypotheses, component fix history

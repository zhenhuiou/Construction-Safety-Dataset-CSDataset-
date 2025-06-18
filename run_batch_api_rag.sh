#!/bin/bash

# Check current Python environment
echo "Current Python environment:"
which python

# Set your OpenAI API key here (if not already set in your environment)
export OPENAI_API_KEY="your_openai_api_key_here"

# Define the input prompt file
INPUT_FILE="prompts_50.csv"

# Run LLM batch inference
echo "Starting LLM batch prediction..."
python main.py --input $INPUT_FILE

# Finished
echo "LLM batch prediction completed!"

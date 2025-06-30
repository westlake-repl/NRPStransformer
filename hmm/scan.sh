#!/bin/bash

# Set default values for parameters (设置参数默认值)
DEFAULT_DATABASE="hmm/core_region/core_region.hmm"
DEFAULT_HMMSCAN_PATH="/usr/bin/hmmscan"
DEFAULT_INPUT_FILE="sequence/sequence.fasta"
DEFAULT_LOG_OUTPUT="hmm/result/log.txt"
DEFAULT_RESULT_OUTPUT="hmm/result/output.txt"

input_file="$1"
file=${input_file:-$DEFAULT_INPUT_FILE}

# Check if the specified input file exists (检查输入文件是否存在)
if [ ! -f "$file" ]; then
    echo "Error: The input file '$file' does not exist."
    exit 1
fi

# Execute hmmscan command with default parameters (使用默认参数执行hmmscan命令)
echo "Running hmmscan..."
"$DEFAULT_HMMSCAN_PATH" --domE 1e-30 --cpu 16 --domtblout "$DEFAULT_RESULT_OUTPUT" "$DEFAULT_DATABASE" "$file" > "$DEFAULT_LOG_OUTPUT"


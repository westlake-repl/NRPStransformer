#!/bin/bash

# Set default values for parameters (设置参数默认值)
DEFAULT_DATABASE="hmm/core_region/core_region.hmm"
DEFAULT_HMMSCAN_PATH="/usr/bin/hmmscan"
DEFAULT_INPUT_FILE="sequence/sequence.fasta"
DEFAULT_LOG_OUTPUT="hmm/result/log.txt"
DEFAULT_RESULT_OUTPUT="hmm/result/output.txt"

# Ask user to input the sequence file path (提示用户输入序列文件路径)
read -p "Please enter the input sequence file path (Default: $DEFAULT_INPUT_FILE): " input_file
file=${input_file:-$DEFAULT_INPUT_FILE}

# Check if the specified input file exists (检查输入文件是否存在)
if [ ! -f "$file" ]; then
    echo "Error: The input file '$file' does not exist."
    exit 1
fi

# Execute hmmscan command with default parameters (使用默认参数执行hmmscan命令)
echo "Running hmmscan..."
"$DEFAULT_HMMSCAN_PATH" --domE 0.01 --cpu 16 "$DEFAULT_DATABASE" "$file" > "$DEFAULT_LOG_OUTPUT"

# Output result processing (输出结果处理)
cat "$DEFAULT_LOG_OUTPUT" | sort -k8n | uniq > "$DEFAULT_RESULT_OUTPUT"

# Completion message (完成提示信息)
echo "Script completed. Please check the output file: $DEFAULT_RESULT_OUTPUT"

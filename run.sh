# /bin/bash

# Make sure current directory is the script's directory
cd "$(dirname "$0")"

# Parameter configuration
DEFAULT_INPUT_FILE="sequence/sequence.fasta"
DEFAULT_RESULT_PATH="result/result.csv"

# Ask user to input the sequence file path
read -p "Please enter the input sequence file path (Default: $DEFAULT_INPUT_FILE): " input_file
input_file=${REPLY:-$DEFAULT_INPUT_FILE}
# Check if the specified input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: The input file '$input_file' does not exist."
    exit 1
fi

# Ask user to input the result file path
read -p "Please enter the result file path (Default: $DEFAULT_RESULT_PATH): "
result_path=${REPLY:-$DEFAULT_RESULT_PATH}


# First Run HMM Scan
if ! ./hmm/scan.sh "$input_file"; then
    echo "Error: HMM scan failed. Please check the input file and parameters."
    exit 1
fi

# Then Parse dbtl
if ! python ./hmm/parse_dbtl.py; then
    echo "Error: Parsing dbtl failed. Please check the output of HMM scan."
    exit 1
fi

# Then generate the domains
if ! python ./hmm/gen_domains.py; then
    echo "Error: Generating domains failed. Please check the parsed dbtl file."
    exit 1
fi

# Finally, run the final AI prediction
if ! python inference.py --result_path "$result_path"; then
    echo "Error: Final AI prediction failed. Please check the input data and parameters."
    exit 1
fi

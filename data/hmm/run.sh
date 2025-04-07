cd /root/project/enzyme_scan

database_file="data/hmm/core_region.hmm"
hmmscan_path="/usr/bin/hmmscan"
file="data/sequence/sequence.fasta"
output_file="data/hmm/output.txt"

$hmmscan_path --domE 0.01 --domtblout "$output_file" --cpu 16 "$database_file" "$file" > log.txt

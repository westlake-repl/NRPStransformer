cd /root/project/ESM_NRPS

database_file="hmm/core_region/core_region.hmm"
hmmscan_path="/usr/bin/hmmscan"
file="sequence/sequence.fasta"
output_log="hmm/result/log.txt"
output_file="hmm/result/output.txt"

$hmmscan_path --domE 0.01 --domtblout "$output_file" --cpu 16 "$database_file" "$file" > "$output_log"

import pandas as pd
import numpy as np

from Bio import SeqIO

def index_fasta(fasta_path):
    id_to_record = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        id_to_record[record.id.split()[0]] = record
    return id_to_record

def prepare_inference(dbtl_csv_path, id2record, output_csv_path):
    df = pd.read_csv(dbtl_csv_path)
    print("Finished reading dbtl")
    
    # 初始化结果列表
    domain_starts = []
    domain_ends = []
    domain_seqs = []
    domain_ids = []
    
    # 按 query_name 分组处理
    for name, group in df.groupby('query_name'):
        # 查找对应的 SeqRecord
        record = id2record.get(name)
        if not record:
            print(f"Warning: Sequence ID '{name}' not found in FASTA file. Skipping.")
            continue
        current_sequence = str(record.seq)  # 获取序列字符串
        
        # 提取该组的 env-from 和 env-to（HMMSCAN的坐标是1-based）
        starts = group['env-from'].astype(int).tolist()
        ends = group['env-to'].astype(int).tolist()
        
        # 验证坐标有效性并转换为0-based索引
        valid_domains = []
        for s, e in zip(starts, ends):
            if 1 <= s <= len(current_sequence) and 1 <= e <= len(current_sequence):
                valid_domains.append((s-1, e))  # 转换为Python的0-based索引
            else:
                print(f"Warning: Invalid coordinates for '{name}' (start={s}, end={e}), sequence length={len(current_sequence)}. Skipping.")
        
        # 分割序列并收集结果
        for s, e in valid_domains:
            domain_seq = current_sequence[s:e]
            domain_seqs.append(domain_seq)
            domain_starts.append(s)
            domain_ends.append(e)
            domain_ids.append(name)
    
    # 转换为numpy数组（与原代码保持一致）
    domain_starts = np.array(domain_starts)
    domain_ends = np.array(domain_ends)
    
    pd.DataFrame({
            "ID": domain_ids,
            "Starts": domain_starts,
            "Ends": domain_ends,
            "Domain": domain_seqs,
        }).to_csv(output_csv_path, index=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract domains from HMM scan results.")
    parser.add_argument('--dbtl_csv_path', type=str, help='Path to the DBTL CSV file.', default='hmm/result/output.csv')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to the FASTA file containing sequences.')
    parser.add_argument('--output_csv_path', type=str, help='Path to save the output CSV file with domains.', default='hmm/result/domains.csv')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dbtl_csv_path = args.dbtl_csv_path
    fasta_path = args.fasta_path
    output_csv_path = args.output_csv_path
    
    id2record = index_fasta(fasta_path)
    prepare_inference(dbtl_csv_path, id2record, output_csv_path)
    print(f"Domains extracted and saved to {output_csv_path}")
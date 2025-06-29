import pandas as pd
import csv

def parse_dbtl(dbtl_path, dbtl_csv_path):
    with open(dbtl_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=' ', skipinitialspace=True)

        table_data = pd.DataFrame(columns=['target_name', 'accession', 'tlen', 'query_name', 'accession', 'qlen', 'E-value', 'score', 'bias', '#', 'of', 'c-Evalue', 'i-Evalue', 'score', 'bias', 'hmm-from', 'hmm-to', 'ali-from', 'ali-to', 'env-from', 'env-to', 'acc', 'description_of_target'])
        for row in csv_reader:
            if not row or row[0].startswith('#'):
                continue
            table_data.loc[len(table_data)] = row
        table_data.to_csv(dbtl_csv_path, index=False)
        
if __name__ == "__main__":
    dbtl_path = 'hmm/result/output.txt'
    dbtl_csv_path = 'hmm/result/output.csv'
    parse_dbtl(dbtl_path, dbtl_csv_path)
    print(f"Parsed {dbtl_path} to {dbtl_csv_path}")
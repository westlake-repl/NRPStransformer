import os
from torch.utils.data import Dataset

os.environ['OPENBLAS_NUM_THREADS'] = '1'

class ProSeqDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_len = df["Domain"].apply(lambda x: len(x)).max()
        self.seqs = list(self.df['Domain'])
        self.inputs = self.tokenizer(list(self.df['Domain']), return_tensors="pt", 
                                    padding="max_length", truncation=True, max_length=self.max_len)
        self.mapped_label = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mapped_label is None:
            label = 0
        else:
            label = self.mapped_label[idx]
        seqs = self.seqs[idx]
        input_ids = self.inputs["input_ids"][idx].squeeze()
        attention_mask = self.inputs["attention_mask"][idx].squeeze()
        return seqs, input_ids, attention_mask, label


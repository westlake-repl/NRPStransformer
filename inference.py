import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from transformers import EsmTokenizer, EsmForSequenceClassification

from data.NRPSDataset import ProSeqDataset
from model.NRPSTransformer import ProFunCla


MODEL_PATH = "./model/esm2_t33_650M_UR50D"
CHECKPOINT_PATH = "checkpoints/epoch=20-val_acc=0.9111-loss=0.0000.ckpt"
NUM_LABELS = 43 
BATCH_SIZE = 8


def main(args):
    val_df = pd.read_csv(args.inference_dataset)

    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
    val_dataset = ProSeqDataset(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=64)

    esm_model = EsmForSequenceClassification.from_pretrained(pretrained_model_name_or_path = MODEL_PATH, num_labels = NUM_LABELS, output_hidden_states=True)
    model = ProFunCla.load_from_checkpoint(model = esm_model,checkpoint_path=CHECKPOINT_PATH, part_result=args.part_result_path, full_result=args.full_result_path)
    
    trainer = pl.Trainer(accelerator="gpu", devices=[0])
    trainer.test(model=model, dataloaders=val_loader)

if __name__ == "__main__":
    import time
    T1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dataset', type=str, default="hmm/result/domains.csv")
    parser.add_argument('--part_result_path', type=str, default="result/part_result.csv")
    parser.add_argument('--full_result_path', type=str, default="result/full_result.csv")
    args = parser.parse_args()
    main(args)
    T2 = time.time()
    print(f"Time: {(T2-T1)}sec")
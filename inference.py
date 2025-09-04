import os
import warnings
import logging

# Suppress warnings as early as possible (before importing libraries that emit warnings)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import argparse
from torch.utils.data import DataLoader
import pandas as pd
from transformers import EsmTokenizer, EsmForSequenceClassification
from transformers.utils import logging as hf_logging

# Reduce third-party library logging verbosity
hf_logging.set_verbosity_error()
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

import pytorch_lightning as pl

from data.NRPSDataset import ProSeqDataset
from model.NRPSTransformer import ProFunCla


MODEL_PATH = "./model/esm2_t33_650M_UR50D"
CHECKPOINT_PATH = "checkpoints/all.ckpt"
NUM_LABELS = 43 
BATCH_SIZE = 8


def main(args):
    val_df = pd.read_csv(args.inference_dataset)

    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
    val_dataset = ProSeqDataset(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=64)

    esm_model = EsmForSequenceClassification.from_pretrained(pretrained_model_name_or_path = MODEL_PATH, num_labels = NUM_LABELS, output_hidden_states=True)
    model = ProFunCla.load_from_checkpoint(model=esm_model, checkpoint_path=CHECKPOINT_PATH, result_path=args.result_path)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model=model, dataloaders=val_loader)
    
    part_result = pd.read_csv(args.result_path)
    final_result = pd.DataFrame(columns=["ID", "Domain", "Top-1(score)", "Top-2(score)", "Top-3(score)"])
    for i in range(len(part_result)):
        domain = part_result.iloc[i]["Domain"]
        top1 = part_result.iloc[i]["Top-1(score)"]
        top2 = part_result.iloc[i]["Top-2(score)"]
        top3 = part_result.iloc[i]["Top-3(score)"]
        id = val_df[val_df["Domain"]==domain]["ID"].iloc[0]  # Assuming 'Domain' is unique in the dataset
        final_result.loc[len(final_result)] = [id, domain, top1, top2, top3]
    final_result.to_csv(args.result_path, index=False)

if __name__ == "__main__":
    import time
    T1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dataset', type=str, default="hmm/result/domains.csv")
    parser.add_argument('--result_path', type=str, default="result/result.csv")
    args = parser.parse_args()
    main(args)
    T2 = time.time()
    print(f"Time: {(T2-T1)}sec")
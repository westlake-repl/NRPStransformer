import torch
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pytorch_lightning as pl
import pandas as pd
from torch.optim.lr_scheduler import StepLR

LABEL_PATH = "model/class_label/labelid2label-43.pt"
CLADE_LABEL_PATH = "model/class_label/labelid2label-17.pt"

def convert_bytes_to_readable(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

class ProFunCla(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, finetune_layer, gama, oversampling, result_path, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        for n, p in self.model.named_parameters():
            if n.startswith("esm.encoder.layer"):
                if int(n.split(".")[3]) <= 33-finetune_layer:
                    p.requires_grad = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.gama = gama
        self.oversampling = oversampling
        # Save the results
        self.result_path = result_path

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def training_step(self, batch, batch_idx):
        seqs, input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        seqs, input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return seqs, outputs, labels
    
    
    def validation_epoch_end(self, outputs):
        if self.num_labels is None:
            return
        logits = torch.cat([x[1].logits for x in outputs], dim=0)
        labels = torch.cat([x[2] for x in outputs], dim=0)
        acc = logits.argmax(dim=1).eq(labels).sum().item() / labels.size(0)
        self.log("val_acc", acc)
        return acc
    
    def test_step(self, batch, batch_idx):
        seqs, input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return seqs, outputs.logits, outputs.hidden_states[-1], labels
    
    def test_epoch_end(self, outputs):
        seqs = [x[0] for x in outputs]
        seqs = [s for seq in seqs for s in seq]
        logits = torch.cat([x[1] for x in outputs], dim=0)
        
        labels = torch.cat([x[3] for x in outputs], dim=0)
        acc = logits.argmax(dim=1).eq(labels).sum().item() / labels.size(0)
        
        prob = torch.nn.functional.softmax(logits, dim=1)
        
        classes = torch.load(LABEL_PATH)
        clade_classes = torch.load(CLADE_LABEL_PATH)
        
        choice = None
        
        try:
            df = pd.DataFrame(prob.cpu().numpy(), columns=classes)
            choice = 43
        except:
            df = pd.DataFrame(prob.cpu().numpy(), columns=clade_classes)
            choice = 17

        part = pd.DataFrame(columns=["Top-1(score)", "Top-2(score)", "Top-3(score)","Domain"])
        for i, seq in enumerate(seqs):
            part.loc[i] = [f"{df.iloc[i].idxmax()}({df.iloc[i].max():.4g})",
                           f"{df.iloc[i].nlargest(2).idxmin()}({df.iloc[i].nlargest(2).min():.4g})",
                           f"{df.iloc[i].nlargest(3).idxmin()}({df.iloc[i].nlargest(3).min():.4g})", seq]

        part.to_csv(self.result_path, index=False)
        
        print(f"Total GPU memory allocated: {convert_bytes_to_readable(torch.cuda.memory_allocated())}")
        return acc
     
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gama)      
        return [optimizer], [scheduler]

import torch
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pytorch_lightning as pl
import pandas as pd
from torch.optim.lr_scheduler import StepLR

LABEL_PATH = "model/class_label/labelid2label-43.pt"


class ProFunCla(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, finetune_layer, gama, oversampling, part_result, full_result, **kwargs):
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
        self.part_result_path = part_result
        self.full_result_path = full_result

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
        self.log("val_acc", acc)
        
        prob = torch.nn.functional.softmax(logits, dim=1)
        
        # Save the results
        classes = torch.load(LABEL_PATH)
        df = pd.DataFrame(prob.cpu().numpy(), columns=classes)

        part = pd.DataFrame(columns=["sequence", "pred", "score"])
        for i, seq in enumerate(seqs):
            part.loc[i] = [seq, df.iloc[i].idxmax(), df.iloc[i].max()]
        part.to_csv(self.part_result_path, index=False)

        whole = pd.DataFrame(columns=["sequence", "pred", "score"]+list(classes))
        for i, seq in enumerate(seqs):
            whole.loc[i] = [seq, df.iloc[i].idxmax(), df.iloc[i].max()] + df.iloc[i].tolist()
        whole.to_csv(self.full_result_path, index=False)
        
        
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gama)      
        return [optimizer], [scheduler]

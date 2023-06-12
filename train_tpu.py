import torch
import wandb
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import (GPT2Config,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from accelerate import Accelerator


class MhcSeqDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep='\t')
        self.alignseq = pd.read_csv('data/mhc.tsv', sep='\t')
        self.mhc = df['Mhc'].tolist()
        self.seq = df['Seq'].tolist()
        self.pred = df['Pred'].tolist()
        return

    def mhc2seq(self, mhc):
        seq = self.alignseq[self.alignseq['Mhc'] == mhc]['Seq'].tolist()
        return seq[0]

    def __len__(self):
        return len(self.mhc)

    def __getitem__(self, item):
        return {
            'mhc': self.mhc2seq(self.mhc[item]),
            'seq': self.seq[item],
            'pred': float(self.pred[item])
        }


class Gpt2ClassificationCollator(object):
    def __init__(self, max_sequence_len):
        self.vocab = ['<pad>', '<mhc>', '</mhc>', '<seq>', '</seq>', '.', '*', 'L', 'A', 'G', 'V', 'E', 'S', 'I', 'K',
                      'R', 'D', 'T', 'P', 'N', 'Q', 'F', 'Y', 'M', 'H', 'C', 'W', 'X', 'U', 'B', 'Z', 'O']
        self.pad_idx = 0
        self.max_sequence_len = max_sequence_len
        return

    def __call__(self, sequences):
        seq = [f"<mhc>{sequence['mhc'].upper()}</mhc><seq>{sequence['seq'].upper()}</seq>" for sequence in sequences]
        labels = [[1 - sequence['pred'], sequence['pred']] for sequence in sequences]
        inputs = self.encode(seq)
        inputs.update({'labels': torch.tensor(labels)})
        return inputs

    def encode(self, sequence):
        result = []
        atten_masks = []
        for seq in sequence:
            ids = []
            while seq:
                found = False
                for i in range(len(self.vocab)):
                    if seq.startswith(self.vocab[i]):
                        ids.append(i)
                        seq = seq[len(self.vocab[i]):]
                        found = True
                        break
                if not found:
                    raise ValueError(f"can't find {seq} in vocab")
            ids = ids[:self.max_sequence_len]
            masks = [1]*len(ids)
            padding_length = self.max_sequence_len - len(ids)
            masks = masks + ([0] * padding_length)
            ids = ids + ([self.pad_idx] * padding_length)
            result.append(ids)
            atten_masks.append(masks)
        return {
            'input_ids': torch.tensor(result),
            'attention_mask': torch.tensor(atten_masks)
        }


def main(use_wandb=True):
    accelerator = Accelerator()
    device = accelerator.device
    print('training on device: ', device)

    epochs = 2
    batch_size = 64
    max_length = 384
    model_name_or_path = 'models/gptneo_3.pt'
    n_labels = 2

    if use_wandb:
        wandb.init(project="gptneo")
        wandb.config.update({
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "model_name_or_path": model_name_or_path,
            "n_labels": n_labels
        })

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=n_labels)
    model_config.vocab_size = 32
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                          config=model_config, ignore_mismatched_sizes=True)
    model.to(device)
    model.resize_token_embeddings(32)
    model.config.pad_token_id = 0
    model.train()
    if use_wandb:
        wandb.watch(model)

    gpt2_classificaiton_collator = Gpt2ClassificationCollator(max_length)
    train_dataset = MhcSeqDataset(path='data/train.tsv')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    train_loss = []
    train_acc = []

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

    for epoch in range(epochs):
        predictions_labels = []
        true_labels = []
        total_loss = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            true_labels += batch['labels'].argmax(axis=-1).cpu().numpy().flatten().tolist()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            if use_wandb:
                wandb.log({"loss": loss.item()})
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = logits.detach().cpu().numpy()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        avg_epoch_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_epoch_loss)
        acc = accuracy_score(true_labels, predictions_labels)
        train_acc.append(acc)
        torch.save(model.state_dict(), f'models/gptneo_{epoch+4}.pt')


if __name__ == '__main__':
    main(True)

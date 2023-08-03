import tqdm
import wandb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from gptneo import GPTConfig, GPTNeo, MhcSeqDataset, Gpt2ClassificationCollator


def train(model, dataset, config, use_wandb=True):
    model.train()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=Gpt2ClassificationCollator(model.config))
    losses = []
    for epoch in range(config['epochs']):
        for i, batch in enumerate(tqdm.tqdm(loader)):
            loss = model(batch['input_ids'].to(model.device), labels=batch['labels'].to(model.device))[0]
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            losses.append(loss.item())
            if use_wandb:
                wandb.log({'loss': loss.item()})

    return model, losses


def evaluate(model, dataset, config, use_wandb=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=Gpt2ClassificationCollator(model.config))
    losses = []
    preds = []
    labels = []
    for i, batch in enumerate(tqdm.tqdm(loader)):
        with torch.no_grad():
            loss, logits = model(batch['input_ids'].to(model.device), labels=batch['labels'].to(model.device))
        losses.append(loss.item())
        preds.append(logits[:, 1].tolist())
        labels.append(batch['labels'][:, 1].tolist())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if use_wandb:
        wandb.log({'loss': np.mean(losses)})
        wandb.log({'roc_auc': roc_auc_score(labels, preds)})
        wandb.log({'pr_auc': average_precision_score(labels, preds)})
    return np.mean(losses), roc_auc_score(labels, preds), average_precision_score(labels, preds)


if __name__ == '__main__':
    model_config = GPTConfig()
    model = GPTNeo(model_config)
    model.to(model.device)
    model, losses = train(model, MhcSeqDataset('data/train.tsv'), {'batch_size': 32, 'epochs': 1})
    torch.save(model.state_dict(), 'model.pt')
    model.load_state_dict(torch.load('model.pt'))
    evaluate(model, MhcSeqDataset('data/test.tsv'), {'batch_size': 32})

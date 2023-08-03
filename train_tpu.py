import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from gptneo import MhcSeqDataset, Gpt2ClassificationCollator, GPTNeo
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator


def main(use_wandb=True):
    accelerator = Accelerator()
    device = accelerator.device
    print('training on device: ', device)

    epochs = 2
    batch_size = 64
    max_length = 384
    model_name_or_path = ''
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

    model = GPTNeo(config={
        'vocab_size': 32,
        'block_size': 384,
        'n_embd': 256,
        'n_layer': 12,
        'n_head': 4,
        'bias': True,
        'dropout': 0.1
    })
    model.to(device)
    model.config.pad_token_id = 0
    model.train()

    if model_name_or_path:
        model.load_state_dict(torch.load(model_name_or_path))
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
        torch.save(model.state_dict(), f'model.pt')


if __name__ == '__main__':
    main(True)

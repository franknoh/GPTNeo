import pandas as pd
import torch
from torch.utils.data import Dataset


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
        inputs.update({
            'labels': torch.tensor(labels),
            'mhc': [sequence['mhc'] for sequence in sequences]
        })
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
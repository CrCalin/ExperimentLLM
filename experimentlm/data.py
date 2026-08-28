import numpy as np
import torch
from torch.utils.data import IterableDataset

class TokenStream(IterableDataset):
    def __init__(self, path, context_length, batch_size, seed=12345):
        self.path, self.context_length, self.batch_size, self.seed = path, context_length, batch_size, seed
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
    def __len__(self): return max(0, (len(self.tokens) - 1) // self.context_length)
    def __iter__(self):
        n = len(self.tokens) - 1; usable = (n // self.context_length) * self.context_length
        starts = np.arange(0, usable, self.context_length); rng = np.random.default_rng(self.seed)
        rng.shuffle(starts)
        for i in range(0, len(starts), self.batch_size):
            s = starts[i:i+self.batch_size]
            if len(s) < self.batch_size: continue
            x = np.stack([self.tokens[j:j+self.context_length] for j in s]).astype(np.int64)
            y = np.stack([self.tokens[j+1:j+self.context_length+1] for j in s]).astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)

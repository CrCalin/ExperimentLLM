import numpy as np
import torch
from torch.utils.data import IterableDataset

class TokenStream(IterableDataset):
    def __init__(self, path, context_length, batch_size, seed=12345, prediction_horizons=1):
        self.path, self.context_length, self.batch_size, self.seed = path, context_length, batch_size, seed
        self.prediction_horizons = prediction_horizons
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.num_sequences = max(0, (len(self.tokens) - 1) // context_length)
        # Generate the order once. This is identical on every iteration and
        # preserves deterministic resume/replay behavior.
        self.starts = np.arange(self.num_sequences, dtype=np.int64) * context_length
        np.random.default_rng(seed).shuffle(self.starts)
        self.batch_starts = self.starts[:(len(self.starts) // batch_size) * batch_size].reshape(-1, batch_size)
        self.offsets = np.arange(context_length, dtype=np.int64)
    def __len__(self): return max(0, (len(self.tokens) - 1) // self.context_length)
    def __iter__(self):
        for starts in self.batch_starts:
            indices = starts[:, None] + self.offsets[None, :]
            # Vectorized indexing replaces one Python/memmap slice per
            # sequence and produces the same [batch, context] layout.
            x = np.take(self.tokens, indices).astype(np.int64, copy=False)
            valid = indices + self.prediction_horizons < len(self.tokens)
            if self.prediction_horizons == 1:
                y = np.take(self.tokens, np.minimum(indices + 1, len(self.tokens) - 1)).astype(np.int64, copy=False)
                y[~valid] = -100
                yield torch.from_numpy(x), torch.from_numpy(y)
            else:
                ys = []
                for h in range(1, self.prediction_horizons + 1):
                    y = np.take(self.tokens, np.minimum(indices + h, len(self.tokens) - 1)).astype(np.int64, copy=False)
                    y[indices + h >= len(self.tokens)] = -100
                    ys.append(y)
                yield torch.from_numpy(x), torch.from_numpy(np.stack(ys))

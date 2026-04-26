import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from torch.utils.data import Dataset


#This dataset class is used to load the cached RVQ tokens from the CSV file and return the codes, token file, segment file, and source file.
class CachedRVQDataset(Dataset):
    def __init__(self, csv_path, debug_n=None):
        df = pd.read_csv(csv_path)
        df = df[df["status"] == "saved"].copy()

        if debug_n is not None:
            df = df.head(debug_n).copy()

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        payload = torch.load(row["token_file"], map_location="cpu")
        codes = payload["codes"].long()   # [K, T]
        return {
                    "codes": codes,
                    "token_file": row["token_file"],
                    "segment_file": row["segment_file"],
                    "source_file": row["source_file"],
                }


#This keeps the original linear layer frozen and adds trainable low-rank adapters only.
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base_layer, nn.Linear) # Ensures to wrap only linear layers.

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze original layer
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))


        # device = base_layer.weight.device
        # dtype = base_layer.weight.dtype

        # self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        # self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))

        # Init: A small random, B zero
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out
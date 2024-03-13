import torch
import torch.nn as nn
import math
from einops import rearrange

# =========== HELPER LAYER ========================================================================
class CheckShape(nn.Module):
    def __init__(self, remark, key=None):
        super().__init__()
        self.remark = remark
        self.key = key
    def forward(self, x):
        if self.remark is not None:
            print(self.remark, x.shape)
        
        out = x
        if self.key is not None:
            out = self.key(x)
        return out

# =========== Encoder Block ========================================================================
class EmbConvBlock(nn.Module):
    def __init__(
        self, 
        in_channel, # 22
        T_kernel_size, # 8,
        emb_size=64,
        hidden_size=64*4
    ):
        super().__init__()
        
        # Input shape: (Series_length, Channel, 1)
        self.liner = nn.Sequential(
            # CheckShape("Before in"),
            # Temporal
            nn.Conv2d(1, hidden_size, kernel_size=[T_kernel_size, 1], padding='same'), 
            # nn.Conv2d(1, emb_size, kernel_size=[T_kernel_size, 1], padding='same', dilation=2), # no warning
            nn.BatchNorm2d(hidden_size), 
            nn.GELU(),
            # CheckShape("1st conv"),
            # Spatial
            nn.Conv2d(hidden_size, emb_size, kernel_size=[1, in_channel], padding='valid'), 
            nn.BatchNorm2d(emb_size), 
            nn.GELU(),
            CheckShape(None, key=lambda x: torch.permute(x, (0, 3, 2, 1))),
            # CheckShape("Emb out"),
        )

    def forward(self, x):
        # Input shape: (N, L, C, 1)
        out = self.liner(x)
        return out
    
# =========== MAIN TRANSFORMER LAYERS ========================================================================
class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len=22, dropout=0.5, add_norm=True):
        super().__init__()
        self.add_norm = add_norm
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(p=dropout)
        self.to_out = nn.LayerNorm(emb_size)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # compute key, query, value vectors
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # compute attention
        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        # add bias
        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # final out
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)

        # Add & Norm
        if self.add_norm:
            return self.LayerNorm(x + out)
        else:
            return out

class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.5, add_norm=True):
        super().__init__()
        self.add_norm = add_norm

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(p=dropout),
            # CheckShape("FC out")
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)

    def forward(self, x):
        out = self.fc_liner(x)
        if self.add_norm:
            return self.LayerNorm(x + out)
        return out
    
class Transformer(nn.Module):
    def __init__(
        self, 
        emb_size=16,
        num_heads=8,
        dropout=0.1,
        hidden_size=256,
        add_norm=True,
        # data related
        in_channel=22,
        seq_length=1024,
    ):
        super().__init__()
        
        # Input shape: (Series_length, Channel, 1)
        self.encoder = EmbConvBlock(in_channel, 8, emb_size, hidden_size=emb_size*4)

        self.transformer = nn.Sequential(
            CheckShape(None, key=lambda x: x.squeeze(1)),
            tAPE(emb_size, dropout=dropout, max_len=seq_length),
            Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=add_norm),
            FeedForward(emb_size, hidden_size, dropout=0.5, add_norm=add_norm),
            # CheckShape("Final Shape")
        )

    def forward(self, x):
        # Input shape: (N, L, C)
        x = x.unsqueeze(1) # (N, 1, L, C)
        emb = self.encoder(x) # (N, 1, L, E)
        out = self.transformer(emb) # (N, L, E)
        return out

if __name__ == '__main__':
    num_sample = 5
    seq_length = 32
    in_channel = 3

    x = torch.rand((num_sample, seq_length, in_channel))
    layer = Transformer(
        emb_size=16,
        num_heads=8,
        dropout=0.1,
        hidden_size=256,
        add_norm=True,
        # data related
        in_channel=in_channel,
        seq_length=seq_length,
    )
    y = layer(x)

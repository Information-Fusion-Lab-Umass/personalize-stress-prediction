import torch
import torch.nn as nn
import math
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE:", DEVICE)

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

class RNNEncoder(nn.Module):
    def __init__(
        self,
        in_channel, # 22
        emb_size=64,
    ):
        super().__init__()

        self.rnn = nn.LSTM(in_channel, emb_size, 1, batch_first=True)
        

    def forward(self, x):
        out, (_, _) = self.rnn(x.squeeze(1))
        return out
    
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

class PersonalizeByPortrait(nn.Module):
    def __init__(
        self, 
        portrait_len=4, 
        num_branch=3,
        in_kernel_size=64,
        out_kernel_size=64,
        tau=1e0,
        num_classes=None
    ):
        super().__init__()

        self.in_len = portrait_len + in_kernel_size
        # self.in_len = portrait_len

        self.branching_param = nn.Sequential(
            nn.Linear(self.in_len, num_branch),
            # nn.Linear(portrait_len, 256),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(256, num_branch),
            nn.LayerNorm(num_branch),
            nn.Sigmoid()
        )

        self.kernel_weight = nn.Linear(self.in_len, int(in_kernel_size*out_kernel_size))
        self.kernel_bias = nn.Linear(self.in_len, out_kernel_size)

        # self.kernel_weight = nn.Sequential(
        #     nn.Linear(self.in_len, 256),
        #     nn.GELU(),
        #     nn.Linear(256, int(in_kernel_size*out_kernel_size)),
        #     nn.Dropout(p=0.5),
        # )
        # self.kernel_bias = nn.Sequential(
        #     nn.Linear(self.in_len, 256),
        #     nn.GELU(),
        #     nn.Linear(256, out_kernel_size),
        #     nn.Dropout(p=0.5),
        # )

        if num_classes is not None:
            self.final_fc = nn.Sequential(
                CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # N, C, L
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(out_kernel_size, num_classes)
            )
        
        self.LayerNorm = nn.LayerNorm(in_kernel_size, eps=1e-5)

        self.tau = tau
        self.num_branch = num_branch
        self.portrait_len = portrait_len
        self.in_kernel_size = in_kernel_size
        self.out_kernel_size = out_kernel_size
        self.num_classes = num_classes
    
    def sampling(self, x, bp):
        # take log
        log_param = torch.log(bp)

        # add noise
        if self.training:
            eps = -torch.log(-torch.log(torch.maximum(torch.Tensor([1e-45]), torch.minimum(torch.Tensor([1-1e-8]), torch.rand(bp.shape))))).to(DEVICE)
        else: # mean around 0.56 - 0.58
            eps = (torch.ones(bp.shape) * 0.57).to(DEVICE)
        # eps = -torch.log(-torch.log(torch.maximum(torch.Tensor([1e-45]), torch.minimum(torch.Tensor([1-1e-8]), torch.rand(bp.shape))))).to(DEVICE)

        log_param = log_param + eps
        
        # calculate probability
        prob = torch.exp(log_param / self.tau)
        prob = torch.nan_to_num(prob, nan=0.0)
        prob = (prob / prob.sum(dim=1, keepdim=True))

        # select branch
        prob = prob.view(-1, 1, self.num_branch, 1)

        # select branch output
        prob = torch.nan_to_num(prob, nan=0.0)
        out = (x * prob).sum(dim=2)
        return out

    def forward(self, x, portrait, latent):
        # x: (N, L, B, H)
        # portrait: (N, P)
        latent_in = torch.cat((torch.mean(torch.mean(x, dim=1), dim=1), portrait), -1)
        # latent_in = portrait
        # output likelihoods and weights
        bp, kw, kb = self.branching_param(latent_in), self.kernel_weight(latent_in), self.kernel_bias(latent_in)

        # select branch
        out = self.sampling(x, bp) # (N, L, H)
        if self.num_branch > 1:
            out = self.sampling(x, bp)
        else:
            out = x.squeeze(2)

        # personal output
        out = torch.matmul(out, kw.view(-1, self.in_kernel_size, self.out_kernel_size)) + kb.unsqueeze(1) # (N, L, H)

        # out = self.LayerNorm(latent + out)
        if self.num_classes is not None:
            out = self.final_fc(out)
        return out

class Branching(nn.Module):
    def __init__(
        self, 
        input_size=64,
        hidden_size=64,
        num_branch=3,
        dropout=0.1
    ):
        super().__init__()

        self.num_branch = num_branch
        
        self.branches = nn.Sequential(
            nn.Linear(input_size, hidden_size*num_branch),
            nn.ReLU(), 
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        # branch forward
        out = self.branches(x)

        # differentiate output from different branches
        N, L, H = out.shape
        if self.num_branch > 1:
            out = out.view(N, L, self.num_branch, -1)
        else:
            out = out.unsqueeze(2)
        return out # (N, L, B, H)
    
class AvgGap(nn.Module):
    def __init__(self, decay=0, task_layer=None):
        super().__init__()
        self.weight_cache = None
        
        self.task_layer = task_layer

        self.avg_liner = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.decay = decay
    
    def forward(self, x, origin=None):
        if self.task_layer is None:
            return self.avg_liner(x)

        N, C, L = x.shape
        # x_norm = torch.norm(x, dim=1, keepdim=True)
        x = x.permute(0, 2, 1).reshape(-1, C)

        with torch.no_grad():
            curr_out = torch.mean(self.task_layer(x), dim=1)
            # relevance = (curr_out - torch.mean(self.task_layer.bias, dim=0)) / torch.norm(torch.mean(self.task_layer.weight, dim=0))
            relevance = (curr_out - torch.mean(self.task_layer.bias, dim=0))

            rel_coef = relevance.view(N, 1, L)
            rel_coef = rel_coef / torch.sum(rel_coef, dim=2, keepdim=True)

        x = x.view(N, C, L)
        self.weight_cache = rel_coef.detach().cpu().numpy()
        if origin is None:
            return torch.sum((x * rel_coef), dim=2)
        return torch.sum((origin * rel_coef), dim=2)

class PersonalLayer(nn.Module):
    def __init__(
        self, 
        emb_size=64,
        dropout=0.5,
        num_branch=7,
        tau=1e1,
        num_classes=1,
        param_update=True,
        group_id=None,
        demo=None,
        seq_length=22,
        num_heads=8,
    ):
        super().__init__()

        self.param_update = param_update
        self.demo = demo
        
        # branch select
        if group_id is None:
            self.params = nn.Parameter(torch.ones(1, num_branch), requires_grad=self.param_update)
        else:
            self.params = nn.Parameter(torch.zeros(1, num_branch), requires_grad=False)
            self.params += 1e-1
            self.params[0][group_id] = 1
            self.params.requires_grad = self.param_update

        # main liner
        self.personal_embed = nn.Linear(emb_size, emb_size)
        self.out_liner = nn.Sequential(
            # # Attention
            # tAPE(emb_size, dropout=dropout, max_len=seq_length),
            # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=True),
            # Normal feed forward
            self.personal_embed,
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=False),
            CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # N, C, L
            # CheckShape("personal final out")
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(emb_size, num_classes)
        )

        self.out_size = num_classes
        self.num_branch = num_branch
        self.tau = tau
        self.group_id = group_id
    
    def sampling(self, x):
        # take log
        log_param = torch.log(self.params)

        # add noise
        if self.param_update:
            if self.training:
                eps = -torch.log(-torch.log(torch.maximum(torch.Tensor([1e-45]), torch.minimum(torch.Tensor([1-1e-8]), torch.rand(self.params.shape))))).to(DEVICE)
            else: # mean around 0.56 - 0.58
                eps = (torch.ones(self.params.shape) * 0.57).to(DEVICE)
            log_param = log_param + eps
        
        # calculate probability
        prob = torch.exp(log_param / self.tau)
        prob = (prob / prob.sum(dim=1, keepdim=True))

        # select branch
        prob = prob.view(-1, 1, self.num_branch, 1)

        # select branch output
        if self.group_id is not None and not self.param_update:
            # out = x[:, :, self.group_id, :]
            out = x[:, self.group_id, :].unsqueeze(0)
        else:
            prob = torch.nan_to_num(prob, nan=0.0)
            out = (x * prob).sum(dim=2)
        # print(x.shape, prob.shape, out.shape)
        return out

    def forward(self, x):
        # sampling
        if self.num_branch > 1:
            out = self.sampling(x)
        else:
            out = x.permute(1, 0, 2)

        # final output
        out = self.out_liner(out)

        if self.demo is not None:
            demos = self.demo.unsqueeze(dim=0)
            out = torch.cat((out, demos), 1)
        return out

class GenericLayer(nn.Module):
    def __init__(
        self, 
        emb_size=64,
        dropout=0.5,
        num_branch=7,
        tau=1e1,
        num_classes=1,
        demo_size=0,
        seq_length=22,
        num_heads=8,
        subjects=list()
    ):
        super().__init__()
        
        # branch select
        # self.personal_branch_params = dict()
        # for s in subjects:
        #     self.personal_branch_params[s] = self.params = nn.Parameter(torch.zeros(1, num_branch), requires_grad=False)
        # self.personal_branch_params = nn.ModuleDict(self.personal_branch_params)
        branch_hidden = 16
        self.demo_branching = nn.Linear(demo_size, branch_hidden)
        self.branch_out = nn.Sequential(
            self.demo_branching,
            nn.ReLU(),
            nn.Linear(branch_hidden, num_branch),
            # nn.Softmax(dim=1),
            CheckShape(None, key=lambda x: x / torch.sum(x, dim=1, keepdim=True)),
            CheckShape(None, key=lambda x: x+0.5)
        )

        # main general liner
        self.out_liner = nn.Sequential(
            # # Attention
            # tAPE(emb_size, dropout=dropout, max_len=seq_length),
            # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=True),
            # Normal feed forward
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=False),
            CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # N, C, L
            # CheckShape("personal final out")
        )

        # # one layer for each subject
        # self.out_liner = dict()
        # for s in subjects:
        #     self.out_liner[s] = nn.Sequential(
        #         # # Attention
        #         # tAPE(emb_size, dropout=dropout, max_len=seq_length),
        #         # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=True),
        #         # Normal feed forward
        #         nn.Linear(emb_size, emb_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=dropout),
        #         # Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout, add_norm=False),
        #         CheckShape(None, key=lambda x: x.permute(0, 2, 1)), # N, C, L
        #         # CheckShape("personal final out")
        #     )
        # self.out_liner = nn.ModuleDict(self.out_liner)

        self.out_size = num_classes
        self.num_branch = num_branch
        self.tau = tau
    
    def sampling(self, x, branch_param):
        '''
        x: (N, L, B, E)
        branch_param: (N, B)
        '''
        # take log
        log_param = torch.log(branch_param)

        # add noise
        if self.training:
            eps = -torch.log(-torch.log(torch.maximum(torch.Tensor([1e-45]), torch.minimum(torch.Tensor([1-1e-8]), torch.rand(branch_param.shape))))).to(DEVICE)
        else: # mean around 0.56 - 0.58
            eps = (torch.ones(branch_param.shape) * 0.57).to(DEVICE)
        log_param = log_param + eps
        
        # calculate probability
        prob = torch.exp(log_param / self.tau)
        prob = (prob / prob.sum(dim=1, keepdim=True))

        # select branch
        prob = prob.view(-1, 1, self.num_branch, 1)

        # select branch output
        prob = torch.nan_to_num(prob, nan=0.0)
        out = (x * prob).sum(dim=2)
        return out

    def forward(self, x, demos, subjects):
        # get branching params
        branch_param = self.branch_out(demos)
        out = self.sampling(x, branch_param)

        # final output
        out = self.out_liner(out)
        # out = torch.stack([self.out_liner[subjects[i]](out[i].unsqueeze(0)).squeeze(0) for i in range(len(subjects))])

        # if self.demo is not None:
        #     demos = self.demo.unsqueeze(dim=0)
        #     out = torch.cat((out, demos), 1)
        return out
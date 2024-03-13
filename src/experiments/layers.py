import torch
from torch import nn

from src.models.layers import *

class autoencoder(nn.Module):
    def __init__(self, AE_type, in_size, hidden_size, num_layers, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.AE_type = AE_type

        if AE_type == 'lstm':
            # LSTM encoder
            self.encoder = nn.LSTM(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
        elif AE_type == 'trans':
            # Transformer encoder
            self.encoder = nn.Sequential(
                Transformer(
                    emb_size=hidden_size, # was 16
                    num_heads=8,
                    dropout=0.1,
                    hidden_size=256,
                    add_norm=True,
                    # data related
                    in_channel=in_size,
                    seq_length=24,
                ),
                CheckShape(None, key=lambda x: (x, (None, None))) # match previous in/out pipeline
            )
        
        # original code below
        self.encoder_act = nn.ReLU()

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=in_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_act = nn.Sigmoid()
    
    def forward(self, x, inds):
        # inds: index of the samples
        # x: (N, ?, D)
        # decoder_out: (N, ?, D)
        # bottle_neck (encoder_out[:, -1, :]): (N, H)

        bottle_neck = torch.zeros(len(inds), self.hidden_size).to(self.device)
        out = list()

        for i in range(len(inds)):
            if self.AE_type == 'trans':
                # encoder forward
                curr_len = x[inds[i]].shape[1]
                encoder_out, (hidden, cell) = self.encoder(x[inds[i]][:, abs(24-curr_len):, :])
            else:
                encoder_out, (hidden, cell) = self.encoder(x[inds[i]])

            encoder_out = self.encoder_act(encoder_out)
            bottle_neck[i] = encoder_out[:, -1, :].squeeze()

            if self.AE_type == 'lstm':
                # decoder forward, tentatively comment out for transformer
                decoder_out, (hidden, cell) = self.decoder(encoder_out)
                out.append(self.decoder_act(decoder_out))

        return out, bottle_neck

class branching(nn.Module):
    def __init__(self, groups, num_branches, device):
        super().__init__()
        self.device = device
        self.num_branches = num_branches
        self.t = 10.0

        self.groups = dict() # map: ids -> group_id
        group_nodes = set()
        for student in groups:
            self.groups[student.split('_')[1]] = groups[student]
            group_nodes.add(groups[student])
        
        self.probabilities = dict() # map: str(ind) -> probability params
        for group in group_nodes:
            self.probabilities[group] = nn.Parameter(torch.ones(self.num_branches, device=self.device), requires_grad=True)
        self.probabilities = nn.ParameterDict(self.probabilities)

    def forward(self, x, ids):
        # x: (N, D)
        # ids: (N, )
        # out: (N, P)
        if self.t > 0.5:
            self.t *= 0.98
            
        N = x.shape[0]

        # fetch probability params
        out = torch.zeros(N, self.num_branches).to(self.device)
        for i in range(len(ids)):
            out[i] = torch.log(self.probabilities[self.groups[ids[i]]])
        
        # gumbel-trick softmax
        eps = -torch.log(-torch.log(torch.rand((N, self.num_branches), device=self.device)))
        out = torch.exp((out + eps) / self.t)

        # # vanilla softmax
        # out = torch.exp(out)

        # final normalize step of softmax
        return out / out.sum(dim=1).view(-1, 1)

class branch_layer(nn.Module):
    def __init__(self, num_branches, in_size, hidden_size, out_size, device):
        super().__init__()
        self.device = device
        self.num_branches = num_branches
        self.out_size = out_size

        self.branches = dict() # map: str(ind) -> sequential
        for i in range(self.num_branches):
            self.branches[str(i)] = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.out_size),
                nn.ReLU()
            )
        self.branches = nn.ModuleDict(self.branches)
    
    def forward(self, x, probabilities):
        # x: (N, D)
        # branch_out: (N, H_B)
        # probabilities: (N, B)
        # out: (N, H_B)
        out = torch.zeros(x.shape[0], self.out_size).to(self.device)
        for i in range(self.num_branches):
            out += self.branches[str(i)](x) * probabilities[:, i].view(-1, 1)
        return out

class out_heads(nn.Module):
    def __init__(self, groups, in_size, hidden_size, out_size, device, transfer_learn=False):
        super().__init__()
        self.device = device
        self.transfer_learn = transfer_learn
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.groups = dict() # map: ids -> group_id
        group_nodes = set()
        for student in groups:
            self.groups[student.split('_')[1]] = groups[student]
            group_nodes.add(groups[student])
        self.num_groups = len([i for i in group_nodes])

        self.out_heads = dict() # map: group_ids -> sequential
        for group in group_nodes:
            self.out_heads[group] = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
            )
        self.out_heads = nn.ModuleDict(self.out_heads)

        self.final_out = dict() # map: group_ids -> sequential
        for group in group_nodes:
            self.final_out[group] = nn.Sequential(
                nn.Linear(hidden_size, out_size),
            )
        self.final_out = nn.ModuleDict(self.final_out)
        
    
    def forward(self, x, ids):
        # x: (N, D)
        # ids: (N, )
        out_size = self.hidden_size if self.transfer_learn else self.out_size
        out = torch.zeros(x.shape[0], out_size).to(self.device)
        for i in range(len(ids)):
            curr_out = self.out_heads[self.groups[ids[i]]](x[i])
            if self.transfer_learn:
                out[i] = curr_out
            else:
                out[i] = self.final_out[self.groups[ids[i]]](curr_out)
        return out

class out_heads_with_generic(nn.Module):
    def __init__(self, groups, in_size, hidden_size, out_size, device, transfer_learn=False):
        super().__init__()
        self.device = device
        self.transfer_learn = transfer_learn
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.groups = dict() # map: ids -> group_id
        group_nodes = set()
        for student in groups:
            self.groups[student.split('_')[1]] = groups[student]
            group_nodes.add(groups[student])
        self.num_groups = len([i for i in group_nodes])

        self.out_heads = dict() # map: group_ids -> sequential
        for group in group_nodes:
            self.out_heads[group] = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
            )
        self.out_heads = nn.ModuleDict(self.out_heads)

        self.final_out = dict() # map: group_ids -> sequential
        for group in group_nodes:
            self.final_out[group] = nn.Sequential(
                nn.Linear(hidden_size, out_size),
            )
        self.final_out = nn.ModuleDict(self.final_out)

        self.generic_out = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )
        
    def forward(self, x, ids):
        # x: (N, D)
        # ids: (N, )
        out_size = self.hidden_size if self.transfer_learn else self.out_size
        out = torch.zeros(x.shape[0], out_size).to(self.device)
        gen_out = torch.zeros(x.shape[0], out_size).to(self.device)
        for i in range(len(ids)):
            gen_out[i] = self.generic_out(x[i])
            curr_out = self.out_heads[self.groups[ids[i]]](x[i])
            if self.transfer_learn:
                out[i] = curr_out
            else:
                out[i] = self.final_out[self.groups[ids[i]]](curr_out)
        return out, gen_out

class personal_head(nn.Module):
    def __init__(self, num_heads, in_size, hidden_size, out_size, device):
        super().__init__()
        self.t = 10.0
        self.device = device
        self.num_heads = num_heads
        self.probabilities = nn.Parameter(torch.ones(num_heads, device=self.device), requires_grad=True)

        self.liner = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )
    
    def forward(self, x):
        # branching
        eps = -torch.log(-torch.log(torch.rand(self.probabilities.shape, device=self.device)))
        prob = torch.exp((torch.log(self.probabilities) + eps) / self.t)
        prob = (prob / prob.sum()).view(1, -1, 1)
        out = (x * prob).sum(dim=1)

        # forward
        return self.liner(out)

# others
class GTS_Linear(nn.Module): # gumbel-trick softmax linear unit
    def __init__(self, in_size, out_size, t=1e1):
        super().__init__()
        self.t = t
        self.W = nn.Parameter(torch.ones(in_size, out_size), requires_grad=True)
    
    def forward(self, x):
        eps = -torch.log(-torch.log(torch.rand(self.W.shape)))
        LOG_W = torch.log(self.W) + eps
        EXP_W = torch.exp(LOG_W / self.t)
        W = EXP_W / EXP_W.sum(dim=1).view(-1, 1)
        return torch.matmul(x, W)
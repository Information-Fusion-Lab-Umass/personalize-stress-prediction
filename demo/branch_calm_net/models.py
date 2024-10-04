import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from branch_calm_net.layers import *

class MultitaskAutoencoder(nn.Module):
    def __init__(self, params, use_covariates=False):
        super().__init__()
        self.transfer_learn = False
        self.with_generic_head = False
        self.use_covariates = use_covariates
        params['shared_in_size'] += params['covariate_size'] if use_covariates else 0
        
        self.params = params
        self.autoencoder = autoencoder(
            AE_type=params['AE'],
            in_size=params['in_size'],
            hidden_size=params['AE_hidden_size'],
            num_layers=params['AE_num_layers'],
            device=params['device'],
        )

        self.branching = branching(
            groups=params['groups'],
            num_branches=params['num_branches'],
            device=params['device'],
        )

        self.branch_layer = branch_layer(
            num_branches=params['num_branches'],
            in_size=params['shared_in_size'],
            hidden_size=params['shared_hidden_size'],
            out_size=params['shared_hidden_size'] // 2,
            device=params['device'],
        )

        self.out_heads = out_heads(
            groups=params['groups'],
            in_size=params['shared_hidden_size'] // 2,
            hidden_size=params['heads_hidden_size'],
            out_size=params['num_classes'],
            device=params['device'],
        )

        # if self.with_generic_head:
        #     self.out_heads = out_heads_with_generic(
        #     groups=params['groups'],
        #     in_size=params['shared_hidden_size'] // 2,
        #     hidden_size=params['heads_hidden_size'],
        #     out_size=params['num_classes'],
        #     device=params['device'],
        # )
        # else:
        #     self.out_heads = out_heads(
        #         groups=params['groups'],
        #         in_size=params['shared_hidden_size'] // 2,
        #         hidden_size=params['heads_hidden_size'],
        #         out_size=params['num_classes'],
        #         device=params['device'],
        #     )

        if self.with_generic_head:
            self.generic_out = nn.Sequential(
                nn.Linear(params['shared_hidden_size'] // 2, params['heads_hidden_size']),
                nn.ReLU(),
                nn.Linear(params['heads_hidden_size'], params['num_classes']),
            )

            # if with prob param
            self.generic_prob = nn.Parameter(torch.ones((1, params['num_branches']), device=params['device']), requires_grad=True)
    
    def forward(self, x, ids, covariate_data=None):
        # autoencoder forward
        AE_out, bottle_neck = self.autoencoder(x)
        if covariate_data is not None and self.use_covariates:
            bottle_neck = torch.cat((bottle_neck, covariate_data), dim=1)
        
        if self.with_generic_head:
            # generic_out = self.generic_out(bottle_neck)

            # if with prob param
            eps = -torch.log(-torch.log(torch.rand((1, self.params['num_branches']), device=self.params['device']))) # gumbel noise
            gen_prob = torch.log(self.generic_prob) # take log
            gen_prob = torch.exp((gen_prob + eps) / self.branching.t) # softmax
            gen_prob /= gen_prob.sum(dim=1).view(-1, 1)
            generic_branch_out = self.branch_layer(bottle_neck, gen_prob) # take input from branches

            generic_out = self.generic_out(generic_branch_out) # final generic out
        
        # select branches
        if not self.transfer_learn:
            probabilities = self.branching(bottle_neck, ids)

            # branches forward
            branch_out = self.branch_layer(bottle_neck, probabilities)
            final_out = self.out_heads(branch_out, ids)
            
            # if not self.with_generic_head:
            #     final_out = self.out_heads(branch_out, ids)
            # else:
            #     final_out, generic_out = self.out_heads(branch_out, ids)

        # for transfer learning
        else:
            B = self.downstream_layers.num_heads
            H = self.params['heads_hidden_size']
            out = torch.zeros(bottle_neck.shape[0], B, H).to(self.autoencoder.device)
            # extract all the existing key of heads
            id_ = [key for key in self.branching.groups]

            # get output from each head one-by-one
            ind = 0
            for i in id_:
                ids = [i for _ in range(bottle_neck.shape[0])]

                probabilities = self.branching(bottle_neck, ids)

                # branches forward
                branch_out = self.branch_layer(bottle_neck, probabilities)
                final_out = self.out_heads(branch_out, ids)
                out[:, ind, :] = final_out
                ind += 1
            
            final_out = self.downstream_layers(out)

        # return
        if not self.with_generic_head:
            return final_out, AE_out
        else:
            return final_out, AE_out, generic_out
    
    def set_transfer_learn(self, downstream_layers):
        self.transfer_learn = True
        self.out_heads.transfer_learn = True
        self.downstream_layers = downstream_layers

# ======== TRAIN SCRIPT ============================================================
    
    def fit(
        self,
        config,
        sample_x,
        sample_y,
        sample_ids,
        sample_cov=None,
        val_x=None,
        val_y=None,
        val_ids=None,
        val_cov=None
    ):
        # Simple training script
        train_loader = DataLoader(
            EXAMPLE_Dataset(sample_x, sample_y, sample_ids, sample_cov),
            batch_size=config['training_params']['batch_size'], 
            shuffle=True,
            # num_workers=4,
        )

        if val_x is not None:
            val_loader = DataLoader(
                EXAMPLE_Dataset(val_x, val_y, val_ids, val_cov),
                batch_size=config['training_params']['batch_size'], 
                shuffle=False,
                # num_workers=4,
            )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config['training_params']['global_lr'],
            weight_decay=config['training_params']['weight_decay'],
        )

        reconstruction_criterion = torch.nn.L1Loss()
        classification_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(config['training_params']['class_weights'], device=config['training_params']['device'])
        )

        train_loss, train_it, val_loss, val_it, iterations = list(), list(), list(), list(), 0
        for e in tqdm(range(config['training_params']['epochs'])):
            # train
            self.train()
            for data_pack in train_loader:
                # forward
                final_out, AE_out = self(data_pack['X'], data_pack['ids'], covariate_data=data_pack['covs'])

                # calc loss
                classification_loss = classification_criterion(final_out, data_pack['y']) * config['training_params']['loss_weight']['beta'] 
                reconstruction_loss = reconstruction_criterion(data_pack['X'], AE_out) *  config['training_params']['loss_weight']['alpha']
                total_loss = classification_loss + reconstruction_loss

                # backpropagation
                self.zero_grad()
                total_loss.backward()
                optimizer.step()
                train_loss.append(total_loss.cpu().detach().item())
                train_it.append(iterations)

                iterations += 1
            
            # val
            if val_x is not None:
                self.eval()
                for data_pack in val_loader:
                    # forward
                    final_out, AE_out = self(data_pack['X'], data_pack['ids'], covariate_data=data_pack['covs'])

                    # calc loss
                    classification_loss = classification_criterion(final_out, data_pack['y']) * config['training_params']['loss_weight']['beta'] 
                    reconstruction_loss = reconstruction_criterion(data_pack['X'], AE_out) *  config['training_params']['loss_weight']['alpha']
                    total_loss = classification_loss + reconstruction_loss
                    val_loss.append(total_loss.cpu().detach().item())
                    val_it.append(iterations)
        
        return train_loss, train_it, val_loss, val_it

# Define dataset/dataloader
class EXAMPLE_Dataset(Dataset):
    def __init__(self, X, y, ids, covs): 
        self.X = X
        self.y = y
        self.ids = ids
        self.covs = covs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "y": self.y[idx],
            "ids": self.ids[idx],
            "covs": self.covs[idx]
        }
    
class LocationMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_liner = nn.Sequential(
            nn.Linear(12, 57),
            nn.Tanh(),
            nn.BatchNorm1d(57),
            nn.Dropout(p=0.35),
            nn.Linear(57, 35),
            nn.Tanh(),
            nn.BatchNorm1d(35),
            nn.Dropout(p=0.25),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.BatchNorm1d(35),
            nn.Dropout(p=0.15),
            nn.Linear(35, 3),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(3),
        )
    
    def forward(self, x):
        return self.fc_liner(x)

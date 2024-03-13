import torch.nn as nn
from torch import optim

from layers import *

'''
Reference github: https://github.com/Navidfoumani/Disjoint-CNN/blob/main/classifiers/Disjoint_CNN.py
Paper from 2021: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9679860&casa_token=KnytDVXnci8AAAAA:yeSw1_D01sDu7gShbipMHtFYW_inauB0tPVP8Vx9yvu4kJTc8g_QtN3NLjUl_81uGWehYIkF3aE

ConvTrans: https://github.com/Navidfoumani/ConvTran/blob/main/Models/model.py
Paper from 2023. https://arxiv.org/pdf/2305.16642v1.pdf
Same 1st and 2nd Authors
'''

class BranchNetMultiTask(nn.Module):
    def __init__(
        self, 
        # model hyper-param:
        in_channel=28, 
        emb_size=64, 
        hidden_size=256,
        num_heads=8,
        num_branch=3,
        dropout=0.5,
        # training specific param:
        num_classes=1, 
        seq_length=22,
        tau=1e0,
        # other:
        subjects=list(),
        personalize=True
    ):
        super().__init__()
        # print("<Check subjects>:", subjects)
        '''
        @param in_size: length of input sequence (e.g. word sequence)
        @param begin_channel: begin_channel
        '''

        # self.loss_func = nn.HuberLoss(delta=1e-8) # 5e-5
        # self.loss_func = lambda x, y: quantile_loss(x, y, quantiles=quantiles)
        # self.quantiles = quantiles

        self.d_cnn_liner = nn.Sequential(
            EmbConvBlock(in_channel, 8, emb_size, hidden_size=emb_size*4),
            CheckShape(None, key=lambda x: x.squeeze(1)),
            tAPE(emb_size, dropout=dropout, max_len=seq_length),
            Attention(emb_size, num_heads, seq_len=seq_length, dropout=dropout),
            FeedForward(emb_size, hidden_size, dropout=0.5),
        )

        self.branching = Branching(
            input_size=emb_size,
            hidden_size=emb_size,
            num_branch=num_branch,
        )

        ###### Personalization
        self.personal = PersonalizeByPortrait(
            portrait_len=4, 
            num_branch=num_branch,
            in_kernel_size=emb_size,
            out_kernel_size=emb_size,
            tau=tau,
            num_classes=num_classes
        )

        # # original version
        # self.personal = dict()
        # if not personalize:
        #     self.out_liner = nn.Sequential(
        #         CheckShape(None, key=lambda x: x.permute(1, 2, 0)), # N, C, L
        #         nn.AdaptiveAvgPool1d(1),
        #         nn.Flatten(),
        #         nn.Linear(emb_size, num_classes)
        #     )
        # for s in subjects:
        #     if personalize:
        #         self.personal[s] = PersonalLayer(
        #             emb_size=emb_size,
        #             dropout=dropout,
        #             num_classes=num_classes,
        #             num_branch=num_branch,
        #             tau=tau, # 1e1
        #             param_update=True,
        #             group_id=None,
        #         )
        #     else:
        #         self.personal[s] = self.out_liner
        # self.personal = nn.ModuleDict(self.personal)
        #####################

        # other variables
        self.emb_size = emb_size
        self.dropout = dropout
        self.out_size = num_classes
        self.num_branch=num_branch
        self.subjects = subjects
        self.seq_length = seq_length

        self.tau = tau
        self.tau_bond = 0.01
    
    def decay_tau(self, decay=0.955):
        self.personal.tau = max(self.tau_bond, self.personal.tau * decay)
        # for sub in self.personal:
        #     self.personal[sub].tau = max(self.tau_bond, self.personal[sub].tau * decay)
        
    def forward(self, x, subjects):
        # x: N x L x C
        N, L, C = x.shape
        x = x.unsqueeze(dim=1) # become (N, 1, L, C)
        latent = self.d_cnn_liner(x) # (N, 1, L, E)

        # branching
        out = self.branching(latent)

        # # personalized output
        # personal_out = torch.stack([self.personal[subjects[i]](out[i]) for i in range(x.shape[0])]).to(DEVICE).squeeze(dim=1)
        # return personal_out

        # with portrait ...
        return self.personal(out, subjects, latent)
    
    def fit(
            self, 
            train_dataset, 
            eval_dataset,
            batch_size=512,
            epochs=50,
            lr=1e-3,
            weight_decay=1e-5,
            step_size=10
        ):
        print("Construct data loader")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

        # construct optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        loss_func = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.997)

        # stored variables
        train_losses, eval_losses = list(), list()
        train_i, eval_i = list(), list()
        iteration = 0
        last_pred = None
        print("Start training")
        for e in tqdm(range(epochs)):
            self.train()
            for data_pack in tqdm(train_loader):
                iteration += 1
                # forward
                # out = self(data_pack["signals"], data_pack["subjects"])
                out = self(data_pack["signals"], data_pack["portrait"])
                loss = loss_func(out, data_pack["labels"])

                # backprop
                optimizer.zero_grad() # clear cache
                loss.backward() # calculate gradient
                # for p in self.parameters(): # addressing gradient vanishing
                #     if p.requires_grad and p.grad is not None:
                #         p.grad = torch.nan_to_num(p.grad, nan=0.0)
                optimizer.step() # update parameters
                scheduler.step() # update learning rate

                # update record
                train_losses.append(loss.detach().cpu().item())
                train_i.append(iteration)

                # update learn-to-branch temperature
                if self.num_branch > 1:
                    if iteration % step_size == 0:
                        self.decay_tau()

            # validation
            if e%2 == 0 or e == epochs-1:
                print("Eval")
                self.eval()
                eval_loss = 0
                total_val_num = 0
                y_preds, y_trues = list(), list()
                with torch.no_grad():
                    for data_pack in tqdm(eval_loader):
                        # out = self(data_pack["signals"], data_pack["subjects"])
                        out = self(data_pack["signals"], data_pack["portrait"])
                        loss = loss_func(out, data_pack["labels"])

                        # update record
                        eval_loss += loss.detach().cpu().item() * len(data_pack["labels"])
                        total_val_num +=  len(data_pack["labels"])

                        # update stored record
                        y_preds += torch.softmax(out, dim=1).detach().cpu().numpy().tolist()
                        y_trues += data_pack["labels"].detach().cpu().numpy().tolist()
                last_pred = {"y_preds": y_preds, "y_trues": y_trues}
                eval_loss /= total_val_num
                eval_losses.append(eval_loss)
                eval_i.append(iteration)
        
        return {
            "train_losses": train_losses, # list
            "train_i": train_i,
            "eval_losses": eval_losses, # list
            "eval_i": eval_i,
            "last_pred": last_pred # dict{list, list}
        }

class Test(Dataset):
    def __init__(self, X, labels, subjects):  
        self.X = X
        self.labels = labels
        self.subjects = subjects

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'signals': torch.tensor(self.X[idx]).float().to(DEVICE),
            'labels': torch.tensor(self.labels[idx]).long().to(DEVICE),
            'subjects': self.subjects[idx],
        }
    
if __name__ == '__main__':
    check_data = torch.rand((5, 22, 28)).to(DEVICE)
    subjects = ["0", "1", "2"]
    ids_ = ["0", "1", "1", "2", "2"]
    labels = [1, 2, 4, 0, 3]

    train_dataset = Test(check_data, labels, ids_)
    eval_dataset = Test(check_data, labels, ids_)

    # check
    print("Input size", check_data.shape)
    model = BranchNetMultiTask(
        in_channel=28, 
        emb_size=64, 
        dropout=0.5,
        num_classes=5, 
        num_branch=1, # 1 is CALM-Net, >1 is Branched CALM-Net
        subjects=subjects,
        personalize=False
    ).to(DEVICE)

    out = model(check_data, ids_)
    print("Output size", out.shape)

    record = model.fit(
        train_dataset, 
        eval_dataset,
        batch_size=8,
        epochs=3,
        lr=1e-3,
        weight_decay=1e-5
    )

    for r in record:
        print(r, record[r])

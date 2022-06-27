import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchdiffeq import odeint_adjoint as odeint

class ODEfunc(nn.Module):
    
    def __init__(self, adj, model_type, num_nodes, latent_dim, nhidden, alpha, embed_dim):
        super(ODEfunc, self).__init__()

        self.model_type = model_type
        self.alpha = alpha
        self.embed_dim = embed_dim
        self.adj = adj
        

        self.A1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        self.A2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        if self.model_type == 'k':
            self.coeff = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))

        # if model_type =='diff':
        #     self.A1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        # if model_type =='AD' or model_type == 'adv':
        #     self.A1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        #     self.A2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, latent_dim)

        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        if self.model_type == 'diff':
            A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A1.T)))
        elif self.model_type == 'adv':
            A_out = F.relu(torch.tanh(self.alpha * (torch.mm(self.A1, self.A2.T) - torch.mm(self.A2, self.A1.T))))
        else: 
            A_out = F.relu(torch.tanh(self.alpha * torch.mm(self.A1, self.A2.T)))

        if self.model_type == 'pre':
            A_out = self.adj
        elif self.model_type == 'k':
            A_out = self.coeff * self.adj
        else:
            A_out = A_out * self.adj


        D_out = torch.diag(A_out.sum(1))
        L = D_out-A_out.T
      

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        out = out.tanh()

        if self.model_type == 'onlyf':
            return out
        elif self.model_type == 'withoutf':
            out = - (torch.matmul(L,x))
            return out
        else:
            out = out - (torch.matmul(L,x))
            return out



class ODEBlock(nn.Module):
    def __init__(self,
                adj,
                model_type,
                num_nodes, 
                in_features,
                horizon,
                alpha,
                embed_dim,
                method='dopri5',
                adjoint=True,
                atol=1e-3,
                rtol=1e-3):
        super(ODEBlock, self).__init__()


        self.odefunc = ODEfunc(adj, model_type, num_nodes, in_features, in_features*4, alpha, embed_dim)
        
        # self.edge_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim*2), requires_grad=True)

        self.horizon = horizon
        self.embed_dim = embed_dim
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.adjoint = adjoint
        # self.fc = nn.Sequential(nn.Linear(in_features, in_features), nn.ELU(), nn.Linear(in_features, out_features))

    def forward(self, x, eval_times=None):
        # temp = torch.zeros(x.shape[-2:]).type_as(x)
        # temp[:,:self.embed_dim*2] = self.edge_embeddings
        # x = torch.cat([x,temp.unsqueeze(0)])

        if eval_times is None:  
            integration_time = torch.linspace(0, self.horizon, self.horizon+1).float().to(x.device)
        else:
            integration_time = eval_times.type_as(x).to(x.device)

        if self.method == 'dopri5':
            out = odeint(self.odefunc, x, integration_time,
                                rtol=self.rtol, atol=self.atol, method=self.method,
                                options={'max_num_steps': 1000})
        else:
            out = odeint(self.odefunc, x, integration_time,
                                rtol=self.rtol, atol=self.atol, method=self.method)        

        # return out[1:,:-1]
        return out[1:]


class Net(nn.Module):
    def __init__(self, args, adj):
        super(Net, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.lag = args.lag
        self.horizon = args.horizon
        self.alpha = args.alpha

        # if args.time_dependence:
            # self.time_control = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1) )
            # self.time_control = nn.Linear(1, 1)

        self.enc = nn.Sequential(nn.Linear(args.input_dim*args.lag, args.hidden_dim),
                                      nn.ReLU())

        self.NADE = ODEBlock(adj, args.model_type, args.num_nodes, args.hidden_dim, args.horizon, self.alpha, args.embed_dim)
        
        self.dec = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(), nn.Linear(args.hidden_dim, args.output_dim))

    def forward(self, X, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D


        X = X.transpose(1,2).reshape(-1,self.num_node, self.input_dim*self.lag)
        X = self.enc(X)

        out = self.NADE(X)
        out = out.permute(1,0,2,3)

        out = self.dec(out)


        return out


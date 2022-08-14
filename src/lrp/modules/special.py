import torch 
import torch.nn as nn 

class SpecialLayer(nn.Module):
    def __init__(self, in_features, activated_features):
        super().__init__()
        self.out_features = in_features
        self.parameter = torch.zeros([in_features,in_features])
        for idx in activated_features:
            self.parameter[idx][idx] = 1
        self.activated = torch.nn.Parameter(self.parameter,requires_grad=False)
        
    def forward(self, x):
        inp = nn.ReLU()(x)
        inp = inp@self.activated
        print(inp)
        output = torch.ones(x.shape[0],self.out_features)
        if torch.sum(inp) == 0 :
            output = torch.zeros_like(output)
            if inp.is_cuda:
                output = output.cuda()
        else:
    #         print(inp)
            if inp.is_cuda:
                output = output.cuda()
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i,j] != 0:
                        output[i] *= inp[i,j]
        ## output은 activated_features들의 곱
        ## Activated된 값은 Relu값?
        return  output

from .utils import construct_incr, construct_rho, clone_layer, keep_conservative

class SpecialLayerLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()

        self.layer = clone_layer(layer)
        self.rho = construct_rho(**rule)
        self.incr = construct_incr(**rule)
        
    def forward(self, Rj, Ai):
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        Rj = torch.ones_like(Rj) * torch.max(Rj) * self.layer.out_features
#         print(Z,Rj)
        S = ((Rj / Z) @ self.layer.activated).data
        return  S
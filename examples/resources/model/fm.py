import torch
import torch.nn as nn
import numpy as np


class FeatureEmbedding(nn.Module):
    def __init__(self, embedding_dims, field_dims):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embedding_dims)
        nn.init.xavier_normal_(self.embedding.weight)
        self.field_dims=field_dims
        
        #self.args=args
        #self.embedding=self.embedding.to(args.device)
        self.offsets=np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
    def forward(self, x):
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FM_Linear(nn.Module):
    def __init__(self, embedding_dims, field_dims):
        super(FM_Linear, self).__init__()
        self.embedding_dims=embedding_dims
        self.linear = nn.Embedding(sum(field_dims), 1)
        #self.bias = nn.Parameter(torch.zeros((1,)))
        #self.bias=nn.Parameter(self.bias.to('cpu'))
        nn.init.xavier_normal_(self.linear.weight)
        #self.linear.weight=nn.Parameter(self.linear.weight.to(args.device))
        self.field_dims=field_dims
        #self.args=args
        #self.linear=self.linear.to(args.device)
        self.offsets=np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
    def forward(self, x):
        # x: batch_size * num_features
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        linear_term=self.linear(x)
        x=torch.sum(linear_term,1)

        return x

class FM_Interaction(nn.Module):
    def __init__(self, embedding_dims, field_dims):
        super(FM_Interaction, self).__init__()
        self.field_dims=field_dims
        self.embedding_dims=embedding_dims
    
    def forward(self, x):
        square_of_sum=torch.sum(x,dim=1)**2
        sum_of_square=torch.sum(x ** 2,dim=1)
        interaction=(square_of_sum-sum_of_square)
        ix=torch.sum(interaction,1,keepdim=True)

        return 0.5*ix
    

class FactorizationMachine(nn.Module):
    def __init__(self, embedding_dims, field_dims):
        super(FactorizationMachine, self).__init__()
        self.linear = FM_Linear(embedding_dims, field_dims)
        self.embedding = FeatureEmbedding(embedding_dims, field_dims)
        self.interaction = FM_Interaction(embedding_dims, field_dims)

    def forward(self, x):
        # x: batch_size * num_features
        return self.linear(x)+self.interaction(self.embedding(x))
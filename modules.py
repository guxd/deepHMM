import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import SOS_ID, EOS_ID

class MLP(nn.Module):
    def __init__(self, input_size, arch, output_size, activation=nn.ReLU(), batch_norm=True, init_w=0.02, discriminator=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_w= init_w

        layer_sizes = [input_size] + [int(x) for x in arch.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and not(discriminator and i==0):# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, self.init_w)
                layer.bias.data.fill_(0)
            except: pass

class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidir, n_layers, dropout=0.5, noise_radius=0.2):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir)==bool
        self.dropout=dropout
        
        self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidir)
        self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad
    
    def forward(self, inputs, input_lens=None, init_h=None, noise=False): 
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.embedding is not None:
            inputs=self.embedding(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, self.dropout, self.training)# dropout
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
        
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        #self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs, init_h) # hids: [b x seq x (n_dir*hid_sz)]  
                                                  # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = h_n.transpose(0,1).contiguous().view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        #if enc.requires_grad:
        #    enc.register_hook(self.store_grad_norm) # store grad norm 
        # norms = torch.norm(enc, 2, 1) # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        # enc = torch.div(enc, norms.unsqueeze(1).expand_as(enc)+1e-5)
        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(enc.size(), device=inputs.device),std=self.noise_radius)
            enc = enc + gauss_noise
            
        return enc, hids


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """
    def __init__(self, z_dim, trans_dim):
        super(GatedTransition, self).__init__()
        self.gate = nn.Sequential( 
            nn.Linear(z_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(z_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim)
        )           
        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """        
        gate = self.gate(z_t_1) # compute the gating function
        proposed_mean = self.proposed_mean(z_t_1) # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean # compute the scale used to sample z_t, using the proposed mean from
        logvar = self.z_to_logvar(self.relu(proposed_mean)) 
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
        z_t = mu + epsilon * torch.exp(0.5 * logvar)    # [batch_sz x z_sz]
        return z_t, mu, logvar 


class PostNet(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim):
        super(PostNet, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh()
        )
        self.h_to_mu = nn.Linear(h_dim, z_dim)
        self.h_to_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, z_t_1, h_x):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """
        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1
        mu = self.h_to_mu(h_combined)
        logvar = self.h_to_logvar(h_combined)
        std = torch.exp(0.5 * logvar)        
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
        z_t = epsilon * std + mu   # [batch_sz x z_sz]
        return z_t, mu, logvar 
    

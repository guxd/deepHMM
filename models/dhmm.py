"""
An implementation of a Deep Markov Model in Pyro based on reference [1].

Adopted from https://github.com/uber/pyro/tree/dev/examples/dmm  
         and https://github.com/clinicalml/structuredinference

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import time
from os.path import exists
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules import GatedTransition, PostNet, Encoder 
from helper import reverse_sequence, sequence_mask


class DHMM(nn.Module):
    """
    The Deep Markov Model
    """
    def __init__(self, config ):
        super(DHMM, self).__init__()
        self.input_dim = config['input_dim']
        self.z_dim = config['z_dim']
        self.emission_dim = config['emission_dim']
        self.trans_dim = config['trans_dim']
        self.rnn_dim = config['rnn_dim']
        self.clip_norm = config['clip_norm']
        
        self.emitter = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.z_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.input_dim)
        )
        self.trans = GatedTransition(self.z_dim, self.trans_dim)
        self.postnet = PostNet(self.z_dim, self.rnn_dim)
        self.rnn = Encoder(None, self.input_dim, self.rnn_dim, False, 1)
                   #nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_dim, nonlinearity='relu', \
                   #batch_first=True, bidirectional=False, num_layers=1)                   

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))    
        
        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas= (config['beta1'], config['beta2']))
            
    def kl_div(self, mu1, logvar1, mu2=None, logvar2=None):
        one = torch.ones(1, device=mu1.device)
        if mu2 is None: mu2=torch.zeros(1, device=mu1.device)
        if logvar2 is None: logvar2=torch.zeros(1, device=mu1.device)
        return torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)             
    
    def infer(self, x, x_rev, x_lens):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        batch_size, _, x_dim = x.size()
        T_max = x_lens.max()
        h_0 = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        
        _, rnn_out = self.rnn(x_rev, x_lens, h_0) # push the observed x's through the rnn;
        rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max), device=x.device) 
        kl_states = torch.zeros((batch_size, T_max), device=x.device)  
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        for t in range(T_max):
            z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            z_t, z_mu, z_logvar = self.postnet(z_prev, rnn_out[:,t,:]) #q(z_t | z_{t-1}, x_{t:T})
            kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)
            rec_losses[:,t] = nn.BCEWithLogitsLoss()(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1))
            z_prev = z_t   
        x_mask = sequence_mask(x_lens)
        x_mask = x_mask.gt(0).view(-1)
        rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
        kl_loss = kl_states.view(-1).masked_select(x_mask).mean()  
        return rec_loss, kl_loss
    
    def train_AE(self, x, x_rev, x_lens, kl_anneal):
        self.rnn.train() # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_anneal*kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()
        
        return {'train_loss_AE':loss.item(), 'train_loss_KL':kl_loss.item()}
    
    def valid(self, x, x_rev, x_lens):
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss
    
    def generate(self, x, x_rev, x_lens):
        """
        generation model p(x_{1:T} | z_{1:T}) p(z_{1:T})
        """
        batch_size, _, x_dim = x.size() # number of time steps we need to process in the mini-batch
        T_max = x_lens.max()
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0)) # set z_prev=z_0 to setup the recursive conditioning in p(z_t|z_{t-1})
        for t in range(1, T_max + 1):
            # sample z_t ~ p(z_t | z_{t-1}) one time step at a time
            z_t, z_mu, z_logvar = self.trans(z_prev) # p(z_t | z_{t-1})
            p_x_t = F.sigmoid(self.emitter(z_t))  # compute the probabilities that parameterize the bernoulli likelihood              
            x_t = torch.bernoulli(p_x_t) #sample observe x_t according to the bernoulli distribution p(x_t|z_t)
            z_prev = z_t



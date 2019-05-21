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
from datetime import datetime
import os
from os.path import exists
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from data_loader import PolyphonicDataset
import models, configs
from helper import get_logger, gVar
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

def save_model(model, epoch):
    ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
    print("saving model to %s..." % ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

def load_model(model, epoch):
    ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
    assert exists(ckpt_path), "epoch misspecified"
    print("loading model from %s..." % ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))


# setup, training, and evaluation
def main(args):
    # setup logging
    log = get_logger(args.log)
    log(args)
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    tb_writer = SummaryWriter("./output/{}/{}/{}/logs/".format(args.model, args.expname, args.dataset)\
                          +timestamp) if args.visual else None
    
    config=getattr(configs, 'config_'+args.model)()
    
    # instantiate the dmm
    model = getattr(models, args.model)(config)
    model = model.cuda()
    if args.reload_from>=0:
        load_model(model, args.reload_from)
        
    train_set=PolyphonicDataset(args.data_path+'train.pkl')
    valid_set=PolyphonicDataset(args.data_path+'valid.pkl')
    test_set=PolyphonicDataset(args.data_path+'test.pkl')

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    for epoch in range(config['epochs']):
            
        train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
        train_data_iter=iter(train_loader)
        n_iters=train_data_iter.__len__()
        
        epoch_nll = 0.0 # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        i_batch=1   
        n_slices=0
        loss_records={}
        while True:            
            try: x, x_rev, x_lens = train_data_iter.next()                  
            except StopIteration: break # end of epoch                 
            x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
            
            if config['anneal_epochs'] > 0 and epoch < config['anneal_epochs']: # compute the KL annealing factor            
                min_af = config['min_anneal']
                kl_anneal = min_af+(1.0-min_af)*(float(i_batch+epoch*n_iters+1)/float(config['anneal_epochs']*n_iters))
            else:            
                kl_anneal = 1.0 # by default the KL annealing factor is unity
            
            loss_AE = model.train_AE(x, x_rev, x_lens, kl_anneal)
            
            epoch_nll += loss_AE['train_loss_AE']
            i_batch=i_batch+1
            n_slices=n_slices+x_lens.sum().item()
            
        loss_records.update(loss_AE)   
        loss_records.update({'epo_nll':epoch_nll/n_slices})
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[Epoch %04d]\t\t(dt = %.3f sec)"%(epoch, epoch_time))
        log(loss_records)
        if args.visual:
            for k, v in loss_records.items():
                tb_writer.add_scalar(k, v, epoch)
        # do evaluation on test and validation data and report results
        if (epoch+1) % args.test_freq == 0:
            save_model(model, epoch)
            test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False, num_workers=1)
            for x, x_rev, x_lens in test_loader: 
                x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
                test_nll = model.valid(x, x_rev, x_lens) / x_lens.sum()
            log("[val/test epoch %08d]  %.8f" % (epoch, test_nll))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data-path', type=str, default='./data/polyphonic/')
    parser.add_argument('--model', type=str, default='DHMM', help='model name')
    parser.add_argument('--dataset', type=str, default='JSBChorales', help='name of dataset. SWDA or DailyDial')
    parser.add_argument('--expname', type=str, default='basic',
                        help='experiment name, for disinguishing different parameter settings')
    parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
    parser.add_argument('--test-freq', type=int, default = 50, help = 'frequency of evaluation in the test set')
    parser.add_argument('-v', '--visual', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()
    
    os.makedirs(f'./output/{args.model}/{args.expname}/{args.dataset}/models', exist_ok=True)
    main(args)

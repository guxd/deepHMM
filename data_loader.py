"""
Data loader logic with two main responsibilities:
(i)  download raw data and process; this logic is initiated upon import
(ii) helper functions for dealing with mini-batches, sequence packing, etc.

Data are taken from

Boulanger-Lewandowski, N., Bengio, Y. and Vincent, P.,
"Modeling Temporal Dependencies in High-Dimensional Sequences: Application to
Polyphonic Music Generation and Transcription"

however, the original source of the data seems to be the Institut fuer Algorithmen
und Kognitive Systeme at Universitaet Karlsruhe.
"""

import os

import numpy as np
import six.moves.cPickle as pickle
import torch
import torch.nn as nn
import torch.utils.data as data

class PolyphonicDataset(data.Dataset):
    def __init__(self, filepath):
        # 1. Initialize file path or list of file names.
        """read training sequences(list of int array) from a pickle file"""
        print("loading data...")
        f= open(filepath, "rb")
        data = pickle.load(f)
        self.seqs = data['sequences']
        self.seqlens = data['seq_lens']
        self.data_len = len(self.seqs)
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        seq=self.seqs[offset].astype('float32')
        rev_seq= seq.copy()
        rev_seq[0:len(seq), :] = seq[(len(seq)-1)::-1, :]
        seq_len=self.seqlens[offset].astype('int64')
        return seq, rev_seq, seq_len

    def __len__(self):
        return self.data_len
   

class SyntheticDataset(data.Dataset):
    def __init__(self, filepath):
        # 1. Initialize file path or list of file names.
        """read training sequences(list of int array) from a pickle file"""
        print("loading data...")
        f= open(filepath, "rb")
        data = pickle.load(f)
        self.seqs = data['sequences']
        self.seqlens = data['seq_lens']
        self.z = data['z']
        self.data_len = len(self.seqs)
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        seq=self.seqs[offset].astype('float32')
        rev_seq= seq.copy()
        rev_seq[0:len(seq), :] = seq[(len(seq)-1)::-1, :]
        seq_len=self.seqlens[offset].astype('int64')
        z = self.z[offset]
        return seq, rev_seq, seq_len, z

    def __len__(self):
        return self.data_len



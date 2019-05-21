import logging
def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)
    return log

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math
import numpy as np

def asHHMMSS(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m /60)
    m -= h *60
    return '%d:%d:%d'% (h, m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asHHMMSS(s), asHHMMSS(rs))


PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]


import torch
from torch.nn import functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def gVar(data):
    tensor=data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    return tensor.to(device)

def sequence_mask(seq_len, max_len=None):
    '''
    Convert sequence lengths to masking vectors
    '''
    if max_len is None:
        max_len = seq_len.data.max()
    batch_size = seq_len.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.long, device=seq_len.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_len_expand = (seq_len.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_len_expand

def reverse_sequence(seqs, seq_lens):
    """
    this function takes a torch mini-batch and reverses each sequence(w.r.t. the temporal axis, i.e. axis=1)
    in contrast to `reverse_sequences_numpy`, this function plays nice with torch autograd
    """
    batch_size, max_seq_len, dim = seqs.size()
    rev_seqs = seqs.new_zeros(seqs.size())
    for b in range(batch_size):
        T = seq_lens[b]
        time_slice = torch.arange(T-1, -1, -1, device=seqs.device)
        rev_seq = torch.index_select(seqs[b, :, :], 0, time_slice)
        rev_seqs[b, 0:T, :] = rev_seq
    return rev_seqs


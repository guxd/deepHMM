"""


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
from observations import jsb_chorales#, musedata, piano_midi_de, nottingham


# this function processes the raw data; in particular it unsparsifies it
def prepare_polyphonic(base_path, name='jsb_chorales', T_max=160, min_note=21, note_range=88):
    print("processing raw polyphonic music data...")
    data = eval(name)(base_path)
    processed = {}
    for split, data_split in zip(['train', 'test', 'valid'], data):
        processed = {}
        n_seqs = len(data_split)
        processed['seq_lens'] = np.zeros((n_seqs), dtype=np.int32)
        processed['sequences'] = np.zeros((n_seqs, T_max, note_range))
        for i in range(n_seqs):            
            seq_len = len(data_split[i])
            processed['seq_lens'][i] = seq_len
            for t in range(seq_len):                
                note_slice = np.array(list(data_split[i][t])) - min_note
                slice_len = len(note_slice)
                if slice_len > 0:
                    processed['sequences'][i, t, note_slice] = np.ones((slice_len))
        f_out = os.path.join(base_path, split+'.pkl')
        pickle.dump(processed, open(f_out, "wb"), pickle.HIGHEST_PROTOCOL)
        print("dumped processed data to %s" % f_out)

        
        
        
        
        
import h5py

def nlinear_trans(z,fxn_params = {}, ns=None): 
    return 2*np.sin(z)+z
def linear_trans(z,fxn_params = {},ns=None): 
    return z+0.05
def linear_obs(z,fxn_params = {},ns=None): 
    return 0.5*z        
params_synthetic = {}
params_synthetic['synthetic9'] = {}
params_synthetic['synthetic9']['trans_fxn']   = linear_trans
params_synthetic['synthetic9']['obs_fxn']     = linear_obs
params_synthetic['synthetic9']['dim_obs']     = 1
params_synthetic['synthetic9']['dim_stoc']    = 1
params_synthetic['synthetic9']['params']      = {}
params_synthetic['synthetic9']['trans_cov']   = 10.
params_synthetic['synthetic9']['trans_cov_full']   = 10.
params_synthetic['synthetic9']['trans_drift'] = 0.05
params_synthetic['synthetic9']['trans_mult']  = 1.
params_synthetic['synthetic9']['obs_cov']     = 20. 
params_synthetic['synthetic9']['obs_cov_full']     = 20. 
params_synthetic['synthetic9']['obs_drift']   = 0. 
params_synthetic['synthetic9']['obs_mult']    = 0.5 
params_synthetic['synthetic9']['init_mu']     = 0.
params_synthetic['synthetic9']['init_cov']    = 1.
params_synthetic['synthetic9']['init_cov_full']    = 1.
params_synthetic['synthetic9']['baseline']    = 'KF' 
params_synthetic['synthetic9']['docstr']      = '$z_t\sim\mathcal{N}(z_{t-1}+0.05,10)$\n$x_t\sim\mathcal{N}(0.5z_t,20)$'
        
        
def prepare_synthetic(base_path, dset):
    assert os.path.exists(base_path),'Directory does not exist: '+base_path
    syntheticDIR = base_path+'/synthetic/'
    if not os.path.exists(syntheticDIR):
        os.mkdir(syntheticDIR)
    fname = syntheticDIR+'/'+dset+'.h5'
    #assert dset in ['synthetic9','synthetic10','synthetic11','synthetic12','synthetic13','synthetic14'] ,'Only synthetic 9/10/11 supported'
    """
    9: linear    ds = 1
    10:nonlinear ds = 1
    11:nonlinear ds = 2 [param estimation]
    Checking scalability of ST-R
    12:linear    ds = 10
    13:linear    ds = 100
    14:linear    ds = 250
    Checking scalability of ST-R - dimz = dimobs
    15:linear    ds = 10
    16:linear    ds = 100
    17:linear    ds = 250
    Checking scalability of ST-R - dimz = dimobs and diagonal weight matrices
    18:linear    ds = 10
    19:linear    ds = 100
    20:linear    ds = 250
    """
    def sampleGaussian(mu, cov):
        assert type(cov) is float or type(cov) is np.ndarray,'invalid type: '+str(cov)+' type: '+str(type(cov))
        return mu + np.random.randn(*mu.shape)*np.sqrt(cov)
    def createDataset(N, T, t_fxn, e_fxn, init_mu, init_cov, trans_cov, obs_cov, model_params, dim_stochastic, dim_obs):
        all_z = []
        z_prev= sampleGaussian(np.ones((N,1,dim_stochastic))*init_mu, init_cov)
        all_z.append(np.copy(z_prev))
        for t in range(T-1):
            z_prev = sampleGaussian(t_fxn(z_prev,fxn_params=model_params), trans_cov) 
            all_z.append(z_prev)
        Z_true= np.concatenate(all_z, axis=1)
        assert Z_true.shape[1]==T,'Expecting T in dim 2 of Z_true'
        X     = sampleGaussian(e_fxn(Z_true, fxn_params = model_params), obs_cov)
        assert X.shape[2]==dim_obs,'Shape mismatch'
        return Z_true, X
    if not np.all([os.path.exists(os.path.join(syntheticDIR,fname+'.h5')) for fname in ['synthetic'+str(i) for i in range(9,21)]]):
        #Create all datasets
        for s in range(9,21):
            if os.path.exists(os.path.join(syntheticDIR,'synthetic'+str(s)+'.h5')):
                print ('Found ',s)
                continue
            print ('Creating: ',s)
            dataset = {}
            transition_fxn = params_synthetic['synthetic'+str(s)]['trans_fxn']
            emission_fxn   = params_synthetic['synthetic'+str(s)]['obs_fxn'] 
            init_mu        = params_synthetic['synthetic'+str(s)]['init_mu']
            init_cov       = params_synthetic['synthetic'+str(s)]['init_cov']
            trans_cov      = params_synthetic['synthetic'+str(s)]['trans_cov']
            obs_cov        = params_synthetic['synthetic'+str(s)]['obs_cov']
            model_params   = params_synthetic['synthetic'+str(s)]['params']
            dim_obs, dim_stoc = params_synthetic['synthetic'+str(s)]['dim_obs'],params_synthetic['synthetic'+str(s)]['dim_stoc']
            if s in [12,13,14,15,16,17,18,19,20]: 
                Ntrain = 1000
                Ttrain = 25 
                Ttest  = 25
            else:
                Ntrain = 5000
                Ttrain = 25 
                Ttest  = 50
            Nvalid = 500
            Ntest  = 500
            #New-
            np.random.seed(1)
            train_Z, train_dataset = createDataset(Ntrain, Ttrain, transition_fxn, emission_fxn,
                                                   init_mu, init_cov, trans_cov, obs_cov,
                                                   model_params, dim_stoc, dim_obs) 
            valid_Z, valid_dataset = createDataset(Nvalid, Ttrain, transition_fxn, emission_fxn,
                                                   init_mu, init_cov, trans_cov, obs_cov,
                                                   model_params, dim_stoc, dim_obs) 
            test_Z,  test_dataset = createDataset(Ntest, Ttest, transition_fxn, emission_fxn,
                                                  init_mu, init_cov, trans_cov, obs_cov,
                                                  model_params, dim_stoc, dim_obs) 
            savefile = syntheticDIR+'/synthetic'+str(s)+'.h5' 
            h5file = h5py.File(savefile,mode='w')
            h5file.create_dataset('train_z', data=train_Z)
            h5file.create_dataset('test_z',  data=test_Z)
            h5file.create_dataset('valid_z', data=valid_Z)
            h5file.create_dataset('train',   data=train_dataset)
            h5file.create_dataset('test',    data=test_dataset)
            h5file.create_dataset('valid',   data=valid_dataset)
            h5file.close()
            print ('Created: ',savefile)
    
    
    
    
# this logic will be initiated upon import
base_path = './data/polyphonic/'
prepare_polyphonic(base_path, 'jsb_chorales')
prepare_synthetic(base_path, 'synthetic9')



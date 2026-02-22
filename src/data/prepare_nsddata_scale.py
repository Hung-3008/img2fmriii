import os
import sys

# Determine the absolute path to the Data directory relative to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '../../Data'))
os.chdir(data_dir)
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-session", "--session",help="Number of Sessions",default=40)
parser.add_argument("-mode", "--mode",help="Normalization mode: scale or zscore",default='scale', choices=['scale','zscore'])
args = parser.parse_args()
sub=int(args.sub)
session=int(args.session)
mode=args.mode
print(f"Processing subject {sub}, {session} sessions, mode: {mode}")
assert sub in [1,2,5,7]

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

stim_order_f = 'nsddata/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)

## Selecting ids for training and test data
sig_train = {}
sig_test = {}
num_trials = session*750
for idx in range(num_trials):
    ''' nsdId as in design csv files'''
    nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx]>1000:
        if nsdId not in sig_train:
            sig_train[nsdId] = []
        sig_train[nsdId].append(idx)
    else:
        if nsdId not in sig_test:
            sig_test[nsdId] = []
        sig_test[nsdId].append(idx)

train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())

roi_dir = 'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
betas_dir = 'nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)

mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()
num_voxel = mask[mask>0].shape[0]

def normalize_within_session(betas, mode):
    if mode == 'scale':
        betas = betas / 2000
        print('Adjusted data (divided by 2000):')
        print(betas.dtype, np.min(betas), np.max(betas), betas.shape)
    elif mode == 'zscore':
        print('z-scoring beta weights within this session...')
        mb = np.mean(betas, axis=0, keepdims=True)
        sb = np.std(betas, axis=0, keepdims=True)
        betas = np.nan_to_num((betas - mb) / np.clip(sb, 1e-8, 10000))
        print(f'min={np.min(betas):.4f}, max={np.max(betas):.4f}, mean={np.mean(betas):.4f}, std={np.std(betas):.4f}')
        print(f'mean(session_mean)={np.mean(mb):.3f}, mean(session_std)={np.mean(sb):.3f}')
    return betas

fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)
for i in range(session):
    beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
    beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)
    betas = beta_f[mask>0].transpose()
    fmri[i*750:(i+1)*750] = normalize_within_session(betas, mode)
    del beta_f
    del betas
    print(i)
    
print("fMRI Data are loaded: ", fmri.shape)

f_stim = h5py.File('nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick'] # Keep as HDF5 dataset, DON'T use [:]

print("Stimuli dataset is opened: ", stim.shape)

num_train, num_test = len(train_im_idx), len(test_im_idx)
vox_dim, im_dim, im_c = num_voxel, 425, 3
fmri_array = np.zeros((num_train,3,vox_dim))
stim_array = np.zeros((num_train,im_dim,im_dim,im_c), dtype=np.uint8) # Pre-allocate with uint8 to save RAM
for i,idx in enumerate(train_im_idx):
    stim_array[i] = stim[idx] # Read only this specific image from disk to RAM
    fmri_array[i] = fmri[sorted(sig_train[idx])]  #[3, voxels]
    # fmri_array[i] = fmri[sorted(sig_train[idx])].mean(0)
    print(i)

os.makedirs('nsd/subj{:02d}'.format(sub), exist_ok=True)
np.save('nsd/subj{:02d}/nsd_train_fmri_{}_sub{}.npy'.format(sub,mode,sub),fmri_array)
np.save('nsd/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub),stim_array)

print("Training data is saved.")

fmri_array = np.zeros((num_test,3,vox_dim))
stim_array = np.zeros((num_test,im_dim,im_dim,im_c), dtype=np.uint8) # Pre-allocate with uint8 to save RAM
for i,idx in enumerate(test_im_idx):
    stim_array[i] = stim[idx] # Read only this specific image from disk to RAM
    fmri_array[i] = fmri[sorted(sig_test[idx])]
    # fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0)
    print(i)

np.save('nsd/subj{:02d}/nsd_test_fmri_{}_sub{}.npy'.format(sub,mode,sub),fmri_array)
np.save('nsd/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub),stim_array)

print("Test data is saved.")


annots_cur = np.load('annots/COCO_73k_annots_curated.npy')

captions_array = np.empty((num_train,5),dtype=annots_cur.dtype)
for i,idx in enumerate(train_im_idx):
    captions_array[i,:] = annots_cur[idx,:]
    print(i)
np.save('nsd/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub),captions_array )
    
captions_array = np.empty((num_test,5),dtype=annots_cur.dtype)
for i,idx in enumerate(test_im_idx):
    captions_array[i,:] = annots_cur[idx,:]
    print(i)
np.save('nsd/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub),captions_array )

print("Caption data are saved.")
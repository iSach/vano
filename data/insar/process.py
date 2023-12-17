import glob
import torch
import numpy as np
import os

def load_data(n_train=4096, res=128):
    # Get base folder of python project
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data128'
    files = glob.glob(f'{data_dir}/*.int', recursive=True)[:n_train]
    print(f"Found {len(files)} files.")
    phi_train = torch.zeros(n_train, res, res).float()
    cos_train = torch.zeros(n_train, res, res).float()
    sin_train = torch.zeros(n_train, res, res).float()
    for i, f in enumerate(files):
        dtype = np.float32
        nline = 128
        nsamp = 128

        with open(f, 'rb') as fn:
            load_arr = np.frombuffer(fn.read(), dtype=dtype)
            img = np.array(load_arr.reshape((nline, nsamp, -1)))

        phi = np.angle(img[:,:,0] + img[:,:,1]*1j)
        phi = torch.tensor(phi[:res, :res])
        
        phi_train[i] = phi
        cos_train[i] = torch.cos(phi)
        sin_train[i] = torch.sin(phi)

    return phi_train, cos_train, sin_train
import os
import sys
import h5py
import torch
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))
from data.transforms import apply_mask, to_tensor
from data.subsample import create_mask_for_mask_type

parser = argparse.ArgumentParser(description='Sample test data from the original test set and save to masked_test_path.')
parser.add_argument('--data_path', type=str, default='/home/alvin/UltrAi/Datasets/raw_datasets/m4raw/multicoil_val', help='path to the original test set')
parser.add_argument('--mask_path', type=str, default='/home/alvin/Masters/EECE_571/PromptMR/data/multicoil_test_mask', help= 'path to the test mask set')
parser.add_argument('--mask_type', type=str, default='equispaced_fraction', help='mask type')
parser.add_argument('--center_fractions', nargs='+', type=float, default=[0.1171875], help='center fraction for the mask')
parser.add_argument('--accelerations', nargs='+', type=int, default=[3], help='acceleration for the mask')

args = parser.parse_args()

data_path = args.data_path
mask_path = args.mask_path
path_prefix = os.path.dirname(data_path)

masked_test_path = join(path_prefix, f'multicoil_val_masked_{args.accelerations[0]}')

assert os.path.exists(data_path), f'{data_path} does not exist!'
assert os.path.exists(mask_path), f'{mask_path} does not exist!'
# assert not os.path.exists(masked_test_path), f'{masked_test_path} already exists!'

if not os.path.exists(masked_test_path):
    os.makedirs(masked_test_path)

test_file_list = os.listdir(data_path)
masks = os.listdir(mask_path)

# generate masked test set and save to masked_test_path
for ti in tqdm(test_file_list):
    # print("ti: ", ti)
    ti_origin_path = join(data_path, ti)
    ti_path = join(masked_test_path, ti)
    # print("ti_origin_path: ", ti_origin_path)
    # print("ti_path: ", ti_path)

    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    with h5py.File(ti_origin_path, 'r') as hf:
        full_kspace = to_tensor(hf['kspace'][()])
        masked_kspace, mask, num_low_frequencies = apply_mask(full_kspace, mask)
        # print("mask: ", mask.shape)
        # print("masked_kspace: ", masked_kspace.shape)
        # convert masked data back to complex
        masked_kspace = torch.view_as_complex(masked_kspace).numpy()
        # print("masked_kspace: ", masked_kspace.shape)        
        # generate masked test set and save to masked_test_path
        with h5py.File(ti_path, 'w') as ht:
            ht.create_dataset('kspace', data=masked_kspace)
            ht.create_dataset('ismrmrd_header', data=hf['ismrmrd_header'])
            ht.create_dataset('mask', data=mask)
            ht.attrs['acceleration'] = args.accelerations[0]
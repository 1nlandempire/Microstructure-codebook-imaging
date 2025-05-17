import torch
import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from torch import nn
from os.path import join
from Model import Model


def normalization(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

print('Loading model ...\n')


net = Model(n_in=3, n_out=24, channels=[32, 64, 128, 256], kernels=128)
device_ids = [0, 1]
model = nn.DataParallel(net)
model.load_state_dict(torch.load('./SUDMEX_60DWIs_40HCs.pth'))
model.eval()
model = model.to('cuda:0')

# dataset_folder
sub_dir = 'SUDMEX_sub-015'


# A B C refer to α β γ, respectively
A_path = join(sub_dir, 'A_60DWIs.nii.gz')
B_path = join(sub_dir, 'B_60DWIs.nii.gz')
C_path = join(sub_dir, 'C_60DWIs.nii.gz')
mask_path = join(sub_dir, 'dwi_mask.nii.gz')
# load
A, affine = load_nifti(A_path, return_img=False)
B, _ = load_nifti(B_path, return_img=False)
C, _ = load_nifti(C_path, return_img=False)
mask, mask_affine = load_nifti(mask_path, return_img=False)
# normalize
A = normalization(A)
B = normalization(B)
C = normalization(C)

# 112 112 50 3
ABC = np.stack([A, B, C], axis=3)
input_ABC = ABC.transpose(2, 3, 0, 1)
mask = np.expand_dims(mask, axis=-1)
mask = mask.transpose(2, 3, 0, 1)

# 50 3 112 112 is the input for the model
print("input size:")
print(input_ABC.shape)


# forward
with torch.no_grad():
    input_ABC = torch.FloatTensor(input_ABC).to('cuda:0')
    codebook, indices, recon = model(input_ABC)
    # to cpu
    codebook = codebook.cpu()
    indices = indices.cpu()
    recon = recon.cpu()
    # * mask
    codebook = codebook * mask
    indices = indices * mask
    recon = recon * mask

# saving results
microstructure = indices.permute(2, 3, 0, 1).numpy()
kernels = codebook.permute(2, 3, 0, 1).numpy()
ABC_recon = recon.permute(2, 3, 0, 1).numpy()
#
print("output size:")
print(microstructure.shape)
print(kernels.shape)
print(ABC_recon.shape)

# denormalize
microstructure[..., 9] *= 0.009
microstructure[..., 10] *= 0.005
microstructure[..., 11] *= 0.005
microstructure[..., 12] *= 0.006
microstructure[..., 13] *= 3
microstructure[..., 15] *= 0.0003
microstructure[..., 16] = (microstructure[..., 16] * 50) ** 4
microstructure[..., 17] *= 0.002
microstructure[..., 18] *= 0.005
microstructure[..., 19] *= 0.005
microstructure[..., 20] *= 0.005
microstructure[..., 21] *= 1.5
microstructure[..., 22] *= 1.5
microstructure[..., 23] *= 1.5

save_path = join(sub_dir, 'output')
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_nifti(join(save_path, 'indices.nii.gz'), microstructure, affine=affine)
save_nifti(join(save_path, 'ABC_recon.nii.gz'), ABC_recon, affine=affine)
save_nifti(join(save_path, 'codebook.nii.gz'), kernels, affine=affine)


import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Vim_CNN(nn.Module):
    def __init__(self, C_in):
        super(Vim_CNN, self).__init__()
        # 1
        self.conv1 = nn.Conv2d(in_channels=C_in, out_channels=C_in*2, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()

        # 2
        self.conv2 = nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        # 3
        self.mamba = Vim_Block(C_in)

    def forward(self, x):

        y1 = self.conv1(x)
        y1 = self.relu1(y1)

        y2 = self.conv2(x)
        y2 = self.relu2(y2)
        y2 = self.conv3(y2)
        y2 = self.relu3(y2)
        #
        y3 = self.mamba(x)

        y = torch.add(y1, torch.cat([y2, y3], dim=1))
        return y




from MultiOrderGatedAggregation import MultiOrderGatedAggregation
from VisionMamba import create_block

class Vim_Block(nn.Module):
    def __init__(self, C, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = create_block(d_model=C)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        hidden_states, residual = self.block(x)
        y = hidden_states.permute(0, 2, 1).view(B, C, H, W)
        return y

class Feature_Refine_n(nn.Module):
    def __init__(self, C_in, n):
        super(Feature_Refine_n, self).__init__()
        self.block1 = MultiOrderGatedAggregation(C_in)
        self.relu1 = nn.ReLU()
        self.mamba1 = Vim_Block(C_in)
        self.conv1 = nn.Conv2d(in_channels=C_in, out_channels=n, kernel_size=1, padding=0)


    def forward(self, x):

        y = self.mamba1(x)
        y = self.block1(y)
        y = self.relu1(y)
        y = self.conv1(y)

        return y



class Model(nn.Module):
    def __init__(self, n_in=3, n_out=24, channels=[32, 64, 128, 256], kernels=128):
        '''
        n_in ---> αβγ
        n_out       ---> output indices
        channels ---> channel number
        '''
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Conv2d(kernel_size=1, in_channels=3, out_channels=channels[0]))
        for ii in range(0, len(channels)):
            layers.append(Vim_CNN(channels[ii]))
        layers.append(Feature_Refine_n(channels[ii] * 2, kernels))

        # n1: αβγ to codebook
        self.n1 = nn.Sequential(*layers)
        # n2: codebook to 24 indices
        layers2 = []
        layers2.append(nn.Conv2d(in_channels=kernels, out_channels=n_out, kernel_size=1, padding=0))
        self.n2 = nn.Sequential(*layers2)

        # n3: codebook to αβγ
        layers3 = []
        layers3.append(nn.Conv2d(in_channels=kernels, out_channels=3, kernel_size=1, padding=0))
        self.n3 = nn.Sequential(*layers3)

    def forward(self, x):
        # αβγ to codebook
        out1 = self.n1(x)

        # codebook to 24 indices
        out2 = self.n2(out1)

        # codebook to αβγ
        out3 = self.n3(out1)

        # codebook, indices, αβγ
        return out1, out2, out3



def main():

    device = 'cuda:0'
    num_channels = 3
    batch_size = 2
    model = Model(3, 24, [32, 64, 128, 256], 128).to(device)
    print(model)

    input = torch.Tensor(torch.rand(batch_size, num_channels, 112, 112)).to(device)
    out1, out2, out3 = model(input)
    print(f'input size {input.shape}:')
    print(f'The output size is {out1.shape}')
    print(f'The output size is {out2.shape}')
    print(f'The output size is {out3.shape}')


if __name__ == "__main__":
    main()




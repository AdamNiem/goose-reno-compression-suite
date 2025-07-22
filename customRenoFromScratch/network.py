import torch
import torch.nn as nn

import torchsparse
from torchsparse import nn as spnn
from torchsparse import SparseTensor

import kit.op as op
from kit.nn import ResNet, FOG, FCG, TargetEmbedding

class Network(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Network, self).__init__() #default pytorch things so that it can set up internal stuff or something
        
        #and assuming this is for downscaling?
        
        #this will be used to convert the discrete voxels into embeddings with 256 dimensions i think
        self.prior_embedding = nn.Embedding(256, channels) #256 nice number
        
        #is this for learning the features to embed? im confused
        self.prior_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )
        
        ###########################
        
        #so im assuming this is for upscaling?

        self.target_embedding = TargetEmbedding(channels) #so this is the end embedding we want?
        
        self.target_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )
        
        ###########################

        self.pred_head_s0 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True), #true for inplace
            nn.Linear(channels, 16),
            nn.Softmax(dim=-1),
        )
        
        self.pred_head_s1_emb = nn.Embedding(16, channels) #so converts 16 dim to channels dim?
        self.pred_head_s1 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True), #true for inplace
            nn.Linear(channels, 16),
            nn.Softmax(dim=-1),
        )
        
        self.channels = channels
        self.fog = FOG() #the secret sauce?!
        self.fcg = FCG()
        
    def forward(self, x):
        N = x.coords.shape[0] #the number of points in the input used for calculating bpp (bits per point)
        
        #get sparse occupancy code list
        #this is still wizardy to me
        data_ls = []
        while True:
            x = self.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone())) #must clone, but why? oh probably so it doesnt make all values added to list the reference to the same processed version of x since x will get processed by fog and then again and again till its down to under 64 points
            if x.coords.shape[0] < 64:
                break #so we keep running it into x 
        data_ls = data_ls[::-1] #reverse the python list same as doing data_ls.reverse() i assume
        scale_list = data_ls
        # data_ls: [(coords, occupancy), (coords, occupancy), ...]
        
        print("scale list / data_ls looks like this: ")
        print(scale_list)
        
        0/0

        total_bits = 0

        for scale_idx in range(len(scale_list)-1):
            curr_coords, curr_occ_codes = scale_list[scale_idx]
            next_coords, next_occ_codes = scale_list[scale_idx+1]
            next_coords, next_occ_codes = op.sort_CF(next_coords, next_occ_codes)

            # embedding prior scale feats
            curr_feats = self.prior_embedding(curr_occ_codes.int()).view(-1, self.channels)  # (N_d, C)
            curr_tensor = SparseTensor(coords=curr_coords, feats=curr_feats)
            curr_tensor = self.prior_resnet(curr_tensor)  # (N_d, C)

            # target embedding
            upscaled_coords, upscaled_feats = self.fcg(curr_coords, curr_occ_codes, curr_tensor.feats)
            upscaled_coords, upscaled_feats = op.sort_CF(upscaled_coords, upscaled_feats)

            upscaled_feats = self.target_embedding(upscaled_feats, upscaled_coords)
            next_tensor = SparseTensor(coords=upscaled_coords, feats=upscaled_feats)
            next_tensor = self.target_resnet(next_tensor)

            # bit-wise two-stage coding
            next_occ_lower = torch.remainder(next_occ_codes, 16)  # lower 4 bits
            next_occ_upper = torch.div(next_occ_codes, 16, rounding_mode='floor')  # upper 4 bits

            lower_prob = self.pred_head_s0(next_tensor.feats)  # (N_{d+1}, 16)
            upper_prob = self.pred_head_s1(next_tensor.feats + self.pred_head_s1_emb(next_occ_lower[:, 0].long()))  # (N_{d+1}, 16)

            lower_prob_gt = lower_prob.gather(1, next_occ_lower.long())  # (N_{d+1}, 1)
            upper_prob_gt = upper_prob.gather(1, next_occ_upper.long())  # (N_{d+1}, 1)

            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(lower_prob_gt + 1e-10), 0, 50))
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(upper_prob_gt + 1e-10), 0, 50))

        bpp = total_bits / N

        return bpp
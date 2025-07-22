import os
import numpy as np
import torch

#these are sort of simple functions i think for just sorting each dimension individually, nvm

def sort_C(coords):
    _, indices = torch.sort(coords[:, 1])  # Sort by x-coordinate
    coords = coords[indices]
    _, indices = torch.sort(coords[:, 2], stable=True)  # Sort by y-coordinate
    coords = coords[indices]
    _, indices = torch.sort(coords[:, 3], stable=True)  # Sort by z-coordinate
    coords = coords[indices]
    _, indices = torch.sort(coords[:, 0], stable=True)  # Sort by batch
    coords = coords[indices]
    return coords

def sort_CF(coords, feats):
    _, indices = torch.sort(coords[:, 1])  # Sort by x-coordinate
    coords = coords[indices]
    feats = feats[indices]
    _, indices = torch.sort(coords[:, 2], stable=True)  # Sort by y-coordinate
    coords = coords[indices]
    feats = feats[indices]
    _, indices = torch.sort(coords[:, 3], stable=True)  # Sort by z-coordinate
    coords = coords[indices]
    feats = feats[indices]
    _, indices = torch.sort(coords[:, 0], stable=True)  # Sort by batch
    coords = coords[indices]
    feats = feats[indices]
    return coords, feats

'''
How it works: 
Essentially sorts each dimension while the stable=True means if the dimension its sorting has two values that are equal then it will choose to preserve there order as it originally was

Example provided from llm for how the above work:
C = [[0, 2, 1, 3], [0, 1, 2, 1], [0, 1, 1, 2]]  # [batch, x, y, z]
F = [[0.5], [0.3], [0.7]]  # features
sort_C(C):
Sort by x (C[:, 1]): [1, 1, 2] → indices [1, 2, 0] → C = [[0, 1, 2, 1], [0, 1, 1, 2], [0, 2, 1, 3]].
Sort by y (C[:, 2]): [2, 1, 1] → indices [2, 0, 1] → C = [[0, 1, 1, 2], [0, 2, 1, 3], [0, 1, 2, 1]].
Sort by z (C[:, 3]): [2, 3, 1] → indices [2, 0, 1] → C = [[0, 1, 2, 1], [0, 1, 1, 2], [0, 2, 1, 3]].
Sort by batch (C[:, 0]): [0, 0, 0] → no change.
Output: C = [[0, 1, 2, 1], [0, 1, 1, 2], [0, 2, 1, 3]].
sort_CF(C, F):
Same sorting steps, but F is reordered with C:
After x-sort: F = [0.3, 0.7, 0.5].
After y-sort: F = [0.7, 0.5, 0.3].
After z-sort: F = [0.3, 0.7, 0.5].
After batch-sort: No change.
Output: C (same as above), F = [0.3, 0.7, 0.5].
'''
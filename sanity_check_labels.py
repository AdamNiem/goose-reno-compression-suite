import numpy as np
import pathlib
import os
from tqdm import tqdm
import sys

np.set_printoptions(threshold=sys.maxsize)

#labels_path = "/scratch/aniemcz/gooseChallengeLabelsCleanMapped"
labels_path = "/scratch/aniemcz/goose-pointcept-decomp-bin/reno/Q_8/labels_challenge"

unique_labels = np.array([])

for (root, dirs, files) in os.walk(labels_path): 
		for file in tqdm(files):
				if pathlib.Path(file).suffix == ".label": 
						print(pathlib.Path(file).stem) 
						# reading a .label file
						label = np.fromfile(os.path.join(root, file), dtype=np.uint32)
						label = label.reshape((-1))

						# extract the semantic and instance label IDs
						sem_label = label & 0xFFFF	# semantic label in lower half
						inst_label = label >> 16	# instance id in upper half
					
						unique_labels = np.unique(np.concatenate((unique_labels, np.unique(sem_label)), axis=0))
				print(unique_labels)


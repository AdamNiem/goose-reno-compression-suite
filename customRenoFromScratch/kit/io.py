import os
import multiprocessing
import numpy as np
from tqdm import tqdm

import open3d as o3d

#So this file is pretty self-explanatory, just want to read in the data into a numpy array

def read_points(file_path):

    if os.path.splitext(file_path)[-1] == ".bin":
        return np.fromfile(file_path, dtype=np.float32).reshape(-1,4)[:, :3] #reshape just as semanticKITTI or goose say to reshape and then only grab the xyz
    
    #now we add functionality so that it can read ascii ply files
    #now could make this faster using something like polars or open3d but will use what they used since it doesnt go into benchmarking encoding time anyway
    ply_file = open(file_path)
    data = []
    #So this part is pretty simple just go through the words separated by spaces of each line, skipping new lines and try to intepret as float
    #headers will always have some text in them so those will fail and give a ValueError so we just skip those lines
    for idx, line in enumerate(ply_file):
        words = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(words):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values) 
    #now with the data we convert to a numpy array so i think each row will be one point so it will be like N x 3
    data = np.array(data)
    coords = data[:, :3] #want to grab only the x y z data not the features like intensity
    return coords

def read_point_clouds(file_path_list):
    print("Loading point clouds")
    with multiprocessing.Pool(64) as p:
        pcs = list(tqdm(p.imap(read_points, file_path_list), total=len(file_path_list)))
    return pcs
    #use the above read points and parallelize it where file_path_list is list of file_paths to read, pcs is list of numpary arrays of the coords data

def save_ply_ascii_geo(coords, filedir):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype("float32"))
    if os.path.exists(filedir):
        os.system("rm " + filedir)
    f = open(filedir, "a+")
    f.writelines(["ply\n", "format ascii 1.0\n"])
    f.write("element vertex " + str(coords.shape[0]) + "\n")
    f.writelines(["property float x\n", "property float y\n", "property float z\n"])
    f.write("end_header\n")
    coords = coords.astype("float32")
    for xyz in coords:
        f.writelines([str(xyz[0]), " ", str(xyz[1]), " ", str(xyz[2]), "\n"])
    f.close()

# -*- coding: utf-8 -*-
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import torch
sep_angle = np.pi / 4  
sep_sin = np.sin(sep_angle) 
total_bins = 4096
step_1 = sep_sin / 2048  
step_2 = (1 - sep_sin) / 2048 
XRD_descriptor = pd.DataFrame(np.zeros((936, total_bins)))
data_path = "./data/XRD_origin.pth"
data = torch.load(data_path)
for i in tqdm(range(936), position=0, leave=True):
    XRD_result = data[i]
    x_angles, y_intensities = XRD_result[1], XRD_result[0]
    x_sin = np.sin(x_angles * np.pi / 360)
    XRD_sequence = np.zeros(total_bins)
    for j, sin_value in enumerate(x_sin):
        intensity = y_intensities[j]
        if sin_value <= sep_sin:
            X_coord = int(np.round(sin_value / step_1))
        else:
            X_coord = int(np.round((sin_value - sep_sin) / step_2) + 2048)
        XRD_sequence[X_coord] += intensity
    
    XRD_descriptor.iloc[i, :] = XRD_sequence
output_path = "./data/XRD_dense_descriptor_new.csv"
XRD_descriptor.to_csv(output_path, index=False, header=False)

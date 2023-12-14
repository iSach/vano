"""
This script is used to process the data from the cahn-hilliard simulation.

The data is available at https://open.bu.edu/handle/2144/43971

The original VANO paper gives few details about their extraction of the data,
and what samples they precisely use. For example, neither the count of samples
is included, nor the part of the dataset. Given the figures shown, my implementation
focuses on the samples from Case 1, and uses uniformly-sampled data from the 37k.
Still, options are available to select only the first samples, or final samples of
each simulation.

Data Structure:
    TODO # what is required for using this script.
"""

"""
⚠: This is a work in progress. The script is not yet functional.
"""

from typing import Union

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt
import cv2

def load_ch(file_path='CH_Dataset_Database.csv'):
    csv = pd.read_csv(file_path, sep=" ")
    final_states = csv.groupby('Cahn_Hilliard_Simulation_Index')
    first_last = final_states['Image_Number'].agg(['first', 'last'])
    first_last = first_last.to_numpy()

    return csv, first_last

def render_sim(first_last, id: int):
    first, last = first_last[id]
    length = last - first + 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'sims/sim_{id}.mp4', fourcc, 15.0, (400, 400), False)
    for i in range(length):
        idx = first + i
        out.write(np.loadtxt(f'case_1_input/Image{idx}.txt', dtype=np.uint8) * 255)
    out.release()

def gen_multires(data):
    pass  # TODO

def select_samples(
    data,
    method: str = 'random'  # 'random', 'final', 'first'
)
    pass  # TODO
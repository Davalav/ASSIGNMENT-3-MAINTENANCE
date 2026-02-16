import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import sys
import os

def inner_join_three_csv(file1, file2, file3, event, merged):
    # Validate file existence
    for f in [file1, file2, file3]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"File not found: {f}")
        
            # Read CSV files
    df1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
    df2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
    df3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')
            

# It seems to have a key --> event
# Maybe not the correct word is merge, rather add together... if that makes sence
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import sys
import os

def inner_join_three_csv(file1, file2, file3, key_column, output_file):
    try:
        # Validate file existence
        for f in [file1, file2, file3]:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"File not found: {f}")
            
                # Read CSV files
        df1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
        df2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
        df3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')

            # Validate key column existence
        for i, df in enumerate([df1, df2, df3], start=1):
            if key_column not in df.columns:
                raise KeyError(f"'{key_column}' not found in file {i}")
            

#This might not merge together, not all columns are the same. 
# And neither does it have and ID or key.

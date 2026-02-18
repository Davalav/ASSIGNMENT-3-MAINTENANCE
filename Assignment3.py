import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



trail1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
trail2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
trail3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')

df = pd.concat([trail1, trail2, trail3], ignore_index = True)
#print(df)
df = df.drop(columns=["start_time", "axle", "tsne_1", "tsne_2"])
print(df)


"""
###ignore_index= True resets the row index after concatenation###
df = concatenate ([trail1, trail2, trail3], ignore_index = True)
            

drop columns


Encode 'event' column (binary classification)
'normal' -> 0, everything else -> 1

"""


# It seems to have a key --> event
# Maybe not the correct word is merge, rather add together... if that makes sence
#Remove start_time, axle, cluster, tsne_1 and tsne_2
#Target column normal, joint X, squat A, etc)
#Why is event considered ground truth? "is labelled by us" - Mohammed Amin Adoul
#tsne represents --> time distributed neural embedded (Visualisation of the Data)
#Normalization after splitting, otherwise it will be leaky data


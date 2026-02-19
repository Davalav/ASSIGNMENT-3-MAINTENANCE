import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Reading CSV files
trail1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
trail2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
trail3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')

# Concatenating the files
df = pd.concat([trail1, trail2, trail3], ignore_index = True)
# Remove Columns
df = df.drop(columns=["start_time", "axle", "cluster", "tsne_1", "tsne_2"], errors='ignore')
#print(df)

# Replacing Event Information
events = []
for x in df["event"]: #For every event we either add a 0 or 1
    if x == "normal":
        events.append(0) #Append adds Zero at the back of the list
    else:
        events.append(1) #Append adds 1 at the back of the list
df["event"] = events #We replace the column information given by the CSV files with one's or zero's from the events list

#Controlling if there is any NaN here
print("How many NaN / column?")
print(df.isnull().sum())


#Scaling the data
scaler = StandardScaler() #Standard Notation
ColumnScaling = df.drop(columns=["event"]) # Drop event, since it shouldn't be transformed it is already bv
scaledColumns = scaler.fit_transform(ColumnScaling)
#NumPy Array, so we need to fix it into normal table again
Table = pd.DataFrame(scaledColumns, columns=ColumnScaling.columns)
#print(Table)
Table["event"] = df["event"].values #Add event back
print(Table)


# It seems to have a key --> event
# Maybe not the correct word is merge, rather add together... if that makes sence
#Remove start_time, axle, cluster, tsne_1 and tsne_2
#Target column normal, joint X, squat A, etc)
#Why is event considered ground truth? "is labelled by us" - Mohammed Amin Adoul
#tsne represents --> time distributed neural embedded (Visualisation of the Data)
#Normalization after splitting, otherwise it will be leaky data


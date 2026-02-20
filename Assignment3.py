import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Import Scaling method
from sklearn.model_selection import train_test_split # Import data splitting method
from sklearn.model_selection import cross_val_score, cross_validate, KFold # Cross-validation methods
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.pipeline import Pipeline # PipeLine method --> Trying to solve the leaky data problem...


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

# Splitting data on to features and Target 
# Features = X and Target = Y
X = df.drop(columns=["event"])
Y = df["event"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train) # fit_transform
X_test_Scaled = scaler.transform(X_test) # transform
# Y doesn't need scaling since it is either 0 or 1 (event) 

#print(X_train)
#print(X_test)

X_train_Scaled = pd.DataFrame(X_train_Scaled, columns=X.columns, index=X_train.index) # index = X_train.index -->Keep same row number as X_train had
X_test_Scaled = pd.DataFrame(X_test_Scaled, columns=X.columns, index=X_test.index) # index = X_test.index --> Keep same row number as X_test had
# We keep the same index placing, so that we know that the information matches with the target

#print(X_train_Scaled)
#print(X_test_Scaled)

svm = SVC(random_state=42)
svm.fit(X_train_Scaled, Y_train)
test_acc = svm.score(X_test_Scaled, Y_test)



svm_NoLeak = Pipeline([
    ("Scaling", scaler),
    ('classifier', svm) 
])


score_cross = cross_val_score(svm_NoLeak, X_train, Y_train, cv=5, scoring='accuracy') 
# Leaky data between all the folds?
# Since it is fitted through all X_train_Scaled --> The scaling has been done with all the five folds included, therefore the data leaks through the five folds.
"""
Previously we had svm, and X_train_scaled --> svm_NoLeak and X_train
"""

print(test_acc)
score_mean = score_cross.mean()
print(score_mean)





"""
The data needs to be splitted before Scaling, because otherwise the scaling will use the whole dataset to scale the data, including the test data.
This can make it look like our model is better than it actually is, but that is only because we have taken the test data into account when we scaled
it.

Stratify? Do we need it?
Stratify makes sure that we have the same proportions of event = 0 and event = 1 in the training data as the test data. So it is probably a good thing.

random_state = 42 means that we make the same split each time we run the code.
"""

#Target column normal, joint X, squat A, etc)
#Why is event considered ground truth? "is labelled by us" - Mohammed Amin Adoul
#tsne represents --> time distributed neural embedded (Visualisation of the Data)
#Normalization after splitting, otherwise it will be leaky data
#---------------> Explain leaky data in report


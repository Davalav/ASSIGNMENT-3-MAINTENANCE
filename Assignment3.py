import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Import Scaling method
from sklearn.model_selection import train_test_split # Import data splitting method
from sklearn.model_selection import cross_val_score, cross_validate, KFold # Cross-validation methods
from sklearn.svm import SVC, LinearSVC # Support Vector Classifier
from sklearn.pipeline import Pipeline # PipeLine method --> Trying to solve the leaky data problem...
from sklearn.feature_selection import mutual_info_classif # Mutual Information
from sklearn.feature_selection import RFE # Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression # LASSO
from sklearn.feature_selection import SelectFromModel


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

X_train_Scaled = pd.DataFrame(X_train_Scaled, columns=X.columns, index=X_train.index) # index = X_train.index -->Keep same row number as X_train had
X_test_Scaled = pd.DataFrame(X_test_Scaled, columns=X.columns, index=X_test.index) # index = X_test.index --> Keep same row number as X_test had
# We keep the same index placing, so that we know that the information matches with the target

svm = SVC(random_state=42)
svm.fit(X_train_Scaled, Y_train)
test_acc = svm.score(X_test_Scaled, Y_test)

svm_NoLeak = Pipeline([ 
    ("Scaling", StandardScaler()),
    ('classifier', SVC(random_state=42)) 
])

score_cross = cross_val_score(svm_NoLeak, X_train, Y_train, cv=5, scoring='accuracy') 
# Leaky data between all the folds?
# Since it is fitted through all X_train_Scaled --> The scaling has been done with all the five folds included, therefore the data leaks through the five folds.

print("------------------------------------------")
print(f"Regular SVM score: {test_acc}")
score_mean = score_cross.mean()
print(f"Cross-Validation score: {score_mean}")
print("------------------------------------------")

# Mutual Information
mi = mutual_info_classif(X_train, Y_train, random_state=42)
mi_table = pd.Series(mi, index=X_train.columns).sort_values(ascending=False).head(4)
mi_TopFeatures = mi_table.index

print(mi_table)

mi_pipe = Pipeline([ 
    ("Scaling", StandardScaler()),
    ('classifier', SVC(random_state=42)) 
])

mi_cross = cross_val_score(mi_pipe, X_train[mi_TopFeatures], Y_train, cv=5, scoring='accuracy') 

mi_pipe.fit(X_train[mi_TopFeatures], Y_train)
mi_test_acc = mi_pipe.score(X_test[mi_TopFeatures], Y_test)

print("------------------------------------------")
print(f"MI Cross-Validation mean: {mi_cross.mean()}")
print(f"MI Test accuracy: {mi_test_acc}")
print("4 Features --> Same accuracy")
print("------------------------------------------")

# Recursive Feature Elimination

RFE_pipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("Selector", RFE(
        estimator=LinearSVC(dual=False, random_state=42), n_features_to_select=4)),
    ("Classifier", SVC(random_state=42))
])

RFE_pipe.fit(X_train, Y_train)
RFE_acc = RFE_pipe.score(X_test,Y_test)
print(f"RFE Test Accuracy: {RFE_acc}")

RFE_cross = cross_val_score(RFE_pipe, X_train, Y_train, cv=5, scoring='accuracy')
print(f"RFE Cross-Validation: {RFE_cross.mean()}")
print("4 Features --> worse than MI (but doing ok)")
print("------------------------------------------")

# LASSO

Lasso_pipe = Pipeline([
    ("Scaler", StandardScaler()),
    ("Selector", SelectFromModel(
     LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=4, #How many features survive
        dual=False, #when n_samples > n_features.
        random_state=42,
        )
    )),
    ("Classifier", SVC(random_state=42))
])
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Lasso_pipe.fit(X_train, Y_train)
Lasso_acc = Lasso_pipe.score(X_test, Y_test)
print(f"Lasso Test Accuracy {Lasso_acc}")

Lasso_cross = cross_val_score(Lasso_pipe, X_train, Y_train, cv=5, scoring='accuracy')
print(f"Lasso Cross-Validation: {Lasso_cross.mean()}")
 # How many features survived LASSO

Lasso_Features = Lasso_pipe.named_steps["Selector"].get_support()
amount = 0
for x in Lasso_Features:
    if x == True:
        amount = amount + 1


#print(Lasso_Features)
print(f"Amount of features: {amount}")


print("------------------------------------------")

#



"""
--> Implement atleast four feature selection algorithms <--
-----------------------------------------------------------
-Filter Methods --> Pearson Correlation or chi-square test ---> Mutual Information (I use)
-Wrapper Methods --> Recursive feature elimination
-Embedded Methods --> LASSO, tree-based models
-----------------------------------------------------------
It would be nice to make a table, where we can compare the different models based on the features we have chosen from each algorithm.
It would also be nice to organize the code in sections, ex all the pipes in one section and all the prints in another.
"""
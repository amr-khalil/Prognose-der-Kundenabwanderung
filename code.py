#!/usr/bin/env python
# coding: utf-8
####Angewandte_Programmierung-Termin 6 ####
# @ Amr Khalil 16.06.2020 21:23


# Pip install xlrd pandas sklearn # Use it if some libraries are missing

# Import Necessary libraries
import pandas as pd # for dataframe

from sklearn.model_selection import train_test_split# Easy way to split the data as traning and test

# Import Machine Learning Algorithms
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


df = pd.read_excel("Kundenabwanderung.xlsx") # import the Excel-sheet
df["Umsatz"].fillna(df["Umsatz"].median(), inplace=True) # remove NaN and place the Median Value of "Umzatz" column 
df["Land"] = df["Land"].factorize()[0] # give numbers as a label instead Lands 
df["Geschlecht"] = df["Geschlecht"].factorize()[0] # give numbers as a label instead M F 
df.head()


# Percentage of Gekuendigt6M 0 = Nicht Gekuendigt 1= Gekuendigt 
df["KundenID"].groupby(df["Gekuendigt6M"]).count()/df["KundenID"].count()*100 

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

X = df.drop(["RowNr","KundenID", "Nachname", "Gekuendigt6M"], axis=1) # Choose the feratures
X = scaler.fit_transform(X) # Scaling the date between 0 and 1 (not necessary but it gives more accuracy)

y = df["Gekuendigt6M"].values # Choose the labels
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.50, random_state=100) # Split the data


# # Logistic Regression Classifier

LR = LogisticRegression(solver="sag", max_iter=100000, multi_class='auto')
LR.fit(X_train, y_train)
prediction = LR.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Linear Support Vector Classifier

LSVC = LinearSVC()
LSVC.fit(X_train, y_train)
prediction = LSVC.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Multinomial Naive Bayes Classifier

MNB = MultinomialNB()
MNB.fit(X_train, y_train)
prediction = MNB.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Bernoulli Naive Bayes Classifier

BNB = BernoulliNB()
BNB.fit(X_train, y_train)
prediction = BNB.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # PCA

# PCA to reduce the fearues into 2 columns 
from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)
pca_model.fit(X_train)
X_train = pca_model.transform(X_train)
X_test = pca_model.transform(X_test)


# # KNN Classifier

K = 70
KNN = KNeighborsClassifier(n_neighbors = K, weights = 'uniform',algorithm = 'auto', leaf_size=50)
KNN.fit(X_train, y_train)
prediction = KNN.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Stochastic Gradient Descent

SGD = SGDClassifier(loss='squared_hinge',  alpha=0.0001, tol=0.1)
SGD.fit(X_train, y_train)
prediction = SGD.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Gradient Boost Classifier

# It's not good with a scaler
GB = GradientBoostingClassifier()
GB.fit(X_train, y_train)
prediction = GB.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Random Forest Classifier

# It's not good with a scaler
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
prediction = RF.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


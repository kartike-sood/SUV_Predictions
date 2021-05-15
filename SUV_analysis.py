from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

#collection of data
data = pd.read_csv("suv_data.csv")
buffer = data.iloc[ : , : 4].copy()
#Analysis of data
# Shown in jupyter notebbok

#Data wrangling
# Based on data analysis
print(data.head(5))

data.drop("User ID", axis = 1, inplace = True)

sex = pd.get_dummies(data["Gender"], drop_first = True)
data = pd.concat([data, sex], axis = 1)
data.drop("Gender", axis = 1, inplace = True)

def l():
    j = 0
    a = np.empty(3)
    for i in range(4):
        if i != 2:
            a[j] = i
            j += 1
    return a

#Training
X = data.iloc[ : , l()]
y = data.loc[: ,"Purchased"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# We are scaling the values of x_train and x_test so that the accuracy of our Classifiers
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test) # fit_transform() is not used because StandardScaler() has already been fitted

clf_log = LogisticRegression()
clf_log.fit(x_train, y_train)

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

y_pred_log = clf_log.predict(x_test)
y_pred_clf = clf.predict(x_test)

print(f"LOGISTIC REGRESSION :\n{classification_report(y_test, y_pred_log, zero_division = 1)}")
print(f"\n\nK-NEIGHBORS CLASSIFIER :\n{classification_report(y_test, y_pred_clf, zero_division = 1)}")

if f1_score(y_test, y_pred_log) >= f1_score(y_test, y_pred_clf):

    choice = "YES"
    while choice == "YES" or choice == "yes" or choice == "Yes":

        num = int(input("Enter the index of the person[0 to 119] : "))
        if num > 119:
            print("############Enter a valid index############")
            continue

        print("The details of this person are as follows :\n\n", buffer.iloc[num, : ], sep = "")

        if clf_log.predict([x_test[num, : ]]) == 1:
            print("This person will purchase the car\n")
        else:
            print("This person will NOT purchase the car\n")

        choice = input("Do you want to test for another case? ")

elif f1_score(y_test, y_pred_log) < f1_score(y_test, y_pred_clf):

    choice = "YES"
    while choice == "YES" or choice == "yes" or choice == "Yes":

        num = int(input("Enter the index of the person[0 to 119] : "))
        if num > 119:
            print("##########Enter a valid index############")
            continue

        print("The details of this person are as follows :\n\n", buffer.iloc[num, :], sep = "")

        if clf.predict([x_test[num, : ]]) == 1:
            print("This person will purchase the car\n")
        else:
            print("This person will NOT purchase the car\n")

        choice = input("Do you want to test for another case? ")
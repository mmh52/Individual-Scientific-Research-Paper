import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def main():
    # data = pd.read_csv('Final-train-dataset-all-without-Unig.csv', sep=',')
    data = pd.read_csv('cleandatafinal.csv', sep=',')
    X = data.iloc[:, 1:]
    Y = data.label
    print(X.shape, Y.shape)  # 2154X3, 300
    # Split dataset into training set and test set
    #keeping 30% data reserved for testing purpose and 70% data will be used to train and form model.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=42)
    print(X_train.shape, Y_train.shape) #1507x3, 1507  
    print(X_test.shape, Y_test.shape)  #647x3, 647
    print("\n\n\n")
    c = []



  
    #Create a svm Classifier with the Linear Function kernel
    model_svm1 = svm.SVC(kernel='linear',decision_function_shape='ovo') 
    #Train the model using the training sets
    model_svm1.fit(X_train, Y_train)
    #Predict the response for test dataset
    c1 = model_svm1.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    accuracy = accuracy_score(Y_test, c1)
    print("Accuracy for SVM with liner function kernel:", accuracy * 100, " %")
    #Compute confusion matrix to evaluate the accuracy of a classification.
    cm = confusion_matrix(Y_test, c1)
    print("Confusion Matrix for SVM (linear)  of Test data\n", cm)
    print("\n\n\n")
   
  
    #Create a svm Classifier with the Radial Basis Function (RBF) kernel
    model_svm2 = svm.SVC(kernel='rbf', C=1.0,gamma='auto',decision_function_shape='ovo') 
    #Train the model using the training sets
    model_svm2.fit(X_train, Y_train)
    #Predict the response for test dataset
    c2 = model_svm2.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    accuracy = accuracy_score(Y_test, c2)
    print("Accuracy for SVM with RBF kernel:", accuracy * 100, " %")
    #Compute confusion matrix to evaluate the accuracy of a classification.
    cm = confusion_matrix(Y_test, c2)
    print("Confusion Matrix for SVM (RBF)  of Test data\n", cm)
    print("\n\n\n")
   


   
    # Create Decision Tree classifer object
    model_dtc = DecisionTreeClassifier(random_state=0)
    #Train Decision Tree Classifer using the training sets
    model_dtc.fit(X_train, Y_train)
    c3 = model_dtc.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    accuracy = accuracy_score(Y_test, c3)
    print("Accuracy for DecisionTree:", accuracy * 100, " %")
    #Compute confusion matrix to evaluate the accuracy of a classification.
    cm = confusion_matrix(Y_test, c3)
    print("Confusion Matrix for DecisionTree of Test data\n", cm)
    print("\n\n\n")





    #Create a Gaussian Naive Bayes algorithm for classification
    model_gnb = GaussianNB()
    #Train the model using the training sets
    model_gnb.fit(X_train, Y_train)
    #Predict the response for test dataset
    c4 = model_gnb.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    accuracy = accuracy_score(Y_test, c4)
    print("Accuracy for Naive Bayes:", accuracy * 100, " %")
    #Compute confusion matrix to evaluate the accuracy of a classification.
    cm = confusion_matrix(Y_test, c4)
    print("Confusion Matrix for Naive Bayes of Test data\n", cm)
    print("\n\n\n")

if __name__ == "__main__":
    main()


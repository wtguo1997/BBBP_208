import pandas as pd
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from xgboost.sklearn import XGBClassifier



def read_data(filename):
    data = pd.read_csv(filename)
    list_data = data.values.tolist()
    info = np.array(list_data)
    info_drugID = info[:,0]
    info_SMILEs = info[:,1]
    info_y = info[:,3]
    info_x = info[:,4:]
    return info_x, info_y

def scale(x_train,x_test):
    scalar = StandardScaler()
    scalar.fit(x_train)
    x_train_scaled = scalar.fit_transform(x_train)
    x_test_scaled = scalar.transform(x_test)
    return x_train_scaled, x_test_scaled


def compute_roc(y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc

def classical(x_train,y_train,x_test,y_test,mltype=str):
    if mltype == "svm":
        svm = SVC(probability=True, gamma=1/x_train.var()/len(x_train[0])) 
        svm.fit(x_train, y_train)
        svm_pred = svm.predict(x_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        print("SVM Accuracy:", svm_accuracy)
        fpr,tpr,roc = compute_roc(y_test, svm.predict_proba(x_test)[:,1])
        plt.plot(fpr, tpr, label='SVM (AUC = {:.2f})'.format(roc))

        return svm_accuracy
    
    if mltype == "lda":
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        lda_pred = lda.predict(x_test)
        fpr, tpr, roc = compute_roc(y_test, lda.predict_proba(x_test)[:,1])
        lda_accuracy = accuracy_score(y_test, lda_pred)
        print("LDA Accuracy:", lda_accuracy)
        plt.plot(fpr, tpr, label='LDA (AUC = {:.2f})'.format(roc))
        return lda_accuracy
    
    if mltype == "dt":
        dt = DecisionTreeClassifier(max_depth=10)
        dt.fit(x_train, y_train)
        dt_pred = dt.predict(x_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        fpr, tpr, roc = compute_roc(y_test, dt.predict_proba(x_test)[:,1])
        print("Decision Tree Accuracy:", dt_accuracy)
        plt.plot(fpr, tpr, label='Decision Tree (AUC = {:.2f})'.format(roc))
        return dt_accuracy  
    
    if mltype == "rf":
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        fpr, tpr, roc = compute_roc(y_test, rf.predict_proba(x_test)[:,1])
        print("Random Forest Accuracy:", rf_accuracy)
        plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(roc))
        return rf_accuracy
    
    if mltype == "lr":
        lr = LogisticRegression(max_iter=10000)
        lr.fit(x_train, y_train)
        lr_pred = lr.predict(x_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        fpr,tpr,roc = compute_roc(y_test, lr.predict_proba(x_test)[:,1])
        print("Logistic Regression Accuracy:", lr_accuracy)
        plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc))
        return lr_accuracy  
    
    if mltype == "gmm":
        gmm = GaussianMixture(n_components=2)
        gmm.fit(x_train, y_train)
        gmm_pred = gmm.predict(x_test)
        gmm_accuracy = accuracy_score(y_test, gmm_pred)
        fpr, tpr, roc = compute_roc(y_test, gmm.predict_proba(x_test)[:,1])
        print("Gaussian Mixture Model Accuracy:", gmm_accuracy)
        plt.plot(fpr, tpr, label='Gaussian Mixture Model (AUC = {:.2f})'.format(roc))
        return gmm_accuracy


    if mltype == "nb":
        nb = MultinomialNB()
        nb.fit(x_train, y_train)
        nb_pred = nb.predict(x_test)
        nb_accuracy = accuracy_score(y_test, nb_pred)
        fpr, tpr, roc = compute_roc(y_test, nb.predict_proba(x_test)[:,1])
        print("Naive Bayes Accuracy:", nb_accuracy)
        plt.plot(fpr, tpr, label='Naive Bayes (AUC = {:.2f})'.format(roc))
        return nb_accuracy
    
    if mltype == "xgb":
        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)
        xgb_pred = xgb.predict(x_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        fpr, tpr, roc = compute_roc(y_test, xgb.predict_proba(x_test)[:,1])
        print("XGBoost Accuracy:", xgb_accuracy)
        plt.plot(fpr, tpr, label='XGBoost (AUC = {:.2f})'.format(roc))
        return xgb_accuracy
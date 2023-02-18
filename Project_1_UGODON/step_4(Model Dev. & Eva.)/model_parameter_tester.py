
#!/usr/bin/python3

import joblib
import numpy
# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score,precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression




def sens_spec_score(y,y_hat):
    TN,TP=0,0
    FN,FP=0,0
    for i in range(len(y)):
        if y.iloc[i]==y_hat[i]:
            if (y_hat[i])==1:
                TP+=1
            else:
                TN+=1
        else:
            if (y_hat[i])==1:
                FP+=1
            else:
                FN+=1
    sensitivity=TP/(TP+FN)
    specificity=TN/(TN+FP)
    precision=TP/(TP+FP)
    antiprecision=TN/(TN+FN)
    return sensitivity,specificity,precision,antiprecision


def draw_check(result,values):
    dt_sens,dt_spec,dt_pre,dt_anpre,dt_f1,dt_acc=[],[],[],[],[],[]
    a=[dt_sens,dt_spec,dt_pre,dt_anpre,dt_f1,dt_acc]

    for i in range(len(result)):
        for j in range(len(a)):
            a[j].append(result[i][j])
    fig, axs = plt.subplots(2,3,figsize = (20,6.5))
    axs[0][0].set_title("sensitivity")    
    axs[1][0].set_title("specificity")
    axs[0][1].set_title("precision")        
    axs[1][1].set_title("antiprecision")
    axs[0][2].set_title("f1_score")        
    axs[1][2].set_title("accuracy")
    for i in range(len(a)):
        axs[i%2][int(i/2)].plot(values,a[i])            #nice reasoning on ...i%2... and ...int(i/2)...


def para_m_n_checker(model,para,p_values,X_train, y_train,X_valid, y_valid,random_state=45): 
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier

    result=[]
    for value in p_values:
        ### create classifier
        if para=='min_samples_split':
            if model=='dt':
                cl_model=tree.DecisionTreeClassifier(min_samples_split=value,random_state=random_state)
            elif model=='rfr' :
                cl_model=RandomForestClassifier(min_samples_split=value,random_state=random_state)
            elif model=='ada' :
                cl_model=AdaBoostClassifier(min_samples_split=value,random_state=random_state)
            elif model=='xg' :
                cl_model=XGBClassifier(min_samples_split=value,random_state=random_state)

        elif para=='n_estimators':
            if model=='dt':
                cl_model=tree.DecisionTreeClassifier(n_estimators=value,random_state=random_state)
            elif model=='rfr':
                cl_model=RandomForestClassifier(n_estimators=value,random_state=random_state)
            elif model=='ada':
                cl_model=AdaBoostClassifier(n_estimators=value,random_state=random_state)
            elif model=='xg':
                cl_model=XGBClassifier(n_estimators=value,random_state=random_state)
                
        elif model=='svm':
            if para=='C':
                cl_model=svm.SVC(kernel='rbf',C=value,random_state=random_state)
            elif para=='gamma':
                cl_model=svm.SVC(kernel='rbf',gamma=value,random_state=random_state)
        elif model=='lr':
            if para=='C':
                cl_model=LogisticRegression(random_state=random_state,C=value)
            elif para=='max_iter':
                cl_model=LogisticRegression(random_state=random_state,max_iter=value)

        cl_model.fit(X_train, y_train)
        y_pred_dt= cl_model.predict(X_valid)
        dt_f1=f1_score(y_valid, y_pred_dt)
        dt_acc=cl_model.score(X_valid, y_valid)
        dt_sens,dt_spec,dt_pre,dt_anpre=sens_spec_score(y_valid, y_pred_dt)
        a=[dt_sens,dt_spec,dt_pre,dt_anpre,dt_f1,dt_acc]
        result.append(a)
    draw_check(result,p_values)
    return result
    

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import table
from sklearn.impute import SimpleImputer
import math
import numpy
from numpy import nan
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
from sklearn.metrics import confusion_matrix


def Calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr


def Get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def DatasetDefinition(consider_unclassified,dataSet):
    
    npdataSet=dataSet.to_numpy(dtype=None)
    listdataSet=[]
    if (consider_unclassified):
      for i in range(len(npdataSet)):
          if pd.isnull(npdataSet[i][34]):
            npdataSet[i][34]="UnClass"
          listdataSet.append(list(npdataSet[i]))
    else:
        for i in range(len(npdataSet)):
          if not pd.isnull(npdataSet[i][34]):
            listdataSet.append(list(npdataSet[i]))
    df=pd.DataFrame(columns=dataSet.columns)
    for i,j in enumerate(dataSet.columns):
      df[j]=[row[i] for row in listdataSet]
    return df

def Get_Label(df):
    y=df['Interaction']
    df=df.drop('Interaction', axis=1)
    df=df.drop('Unnamed: 0', axis=1)
    return df,y,list(set(y)).sort()


def Preprocessing(df):
    df['s_ss8'] = SimpleImputer(missing_values=np.NaN ,strategy='most_frequent').fit_transform(np.array(df['s_ss8'].values)[:, np.newaxis])
    df['s_rsa'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['s_rsa'].values)[:, np.newaxis])
    df['s_up'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['s_up'].values)[:, np.newaxis])
    df['s_down'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['s_down'].values)[:, np.newaxis])
    df['s_phi'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['s_phi'].values)[:, np.newaxis])
    df['s_psi'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['s_psi'].values)[:, np.newaxis])
    df['s_ss3'] = SimpleImputer(missing_values=np.NaN ,strategy='most_frequent').fit_transform(np.array(df['s_ss3'].values)[:, np.newaxis])
    df['t_ss8'] = SimpleImputer(missing_values=np.NaN ,strategy='most_frequent').fit_transform(np.array(df['t_ss8'].values)[:, np.newaxis])
    df['t_rsa'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['t_rsa'].values)[:, np.newaxis])
    df['t_up'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['t_up'].values)[:, np.newaxis])
    df['t_down'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['t_down'].values)[:, np.newaxis])
    df['t_phi'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['t_phi'].values)[:, np.newaxis])
    df['t_psi'] = SimpleImputer(missing_values=np.NaN ,strategy='mean').fit_transform(np.array(df['t_psi'].values)[:, np.newaxis])
    df['t_ss3'] = SimpleImputer(missing_values=np.NaN ,strategy='most_frequent').fit_transform(np.array(df['t_ss3'].values)[:, np.newaxis])
    return df


def LabelEncoder(df,y):
    pdb_idL = preprocessing.LabelEncoder()
    pdb_idL.fit(df['pdb_id'])
    df['pdb_id']=pdb_idL.transform(df['pdb_id'])
    s_ins = preprocessing.LabelEncoder()
    s_ins.fit(df['s_ins'])
    df['s_ins']=s_ins.transform(df['s_ins'])
    s_resn = preprocessing.LabelEncoder()
    s_resn.fit(df['s_resn'])
    df['s_resn']=s_resn.transform(df['s_resn'])
    s_ch = preprocessing.LabelEncoder()
    s_ch.fit(df['s_ch'])
    df['s_ch']=s_ch.transform(df['s_ch'])
    s_ss8	 = preprocessing.LabelEncoder()
    s_ss8.fit(df['s_ss8'])
    df['s_ss8']=s_ss8.transform(df['s_ss8'])
    s_ss3 = preprocessing.LabelEncoder()
    s_ss3.fit(df['s_ss3'])
    df['s_ss3']=s_ss3.transform(df['s_ss3'])
    t_ch = preprocessing.LabelEncoder()
    t_ch.fit(df['t_ch'])
    df['t_ch']=t_ch.transform(df['t_ch'])
    t_ch = preprocessing.LabelEncoder()
    t_ch.fit(df['t_ch'])
    df['t_ch']=t_ch.transform(df['t_ch'])
    t_ins = preprocessing.LabelEncoder()
    t_ins.fit(df['t_ins'])
    df['t_ins']=t_ins.transform(df['t_ins'])
    t_resn = preprocessing.LabelEncoder()
    t_resn.fit(df['t_resn'])
    df['t_resn']=t_resn.transform(df['t_resn'])
    t_ss8 = preprocessing.LabelEncoder()
    t_ss8.fit(df['t_ss8'])
    df['t_ss8']=t_ss8.transform(df['t_ss8'])
    t_ss3 = preprocessing.LabelEncoder()
    t_ss3.fit(df['t_ss3'])
    df['t_ss3']=t_ss3.transform(df['t_ss3'])
    yy = preprocessing.LabelEncoder()
    yy.fit(y)
    y=yy.transform(y)
    y_bin = np_utils.to_categorical(y)
    return df,y,y_bin

def OverSample(df,y,col):
    del col[0]
    del col[-1]
    X=df.to_numpy()
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    dfX=pd.DataFrame(columns=col)

    for i,j in enumerate(dfX.columns):
  
        dfX[j]=[row[i] for row in X]

    return X,y,np_utils.to_categorical(y)

def PrintLossAccuracy(hist):
    loss_train = hist.history['loss']
    loss_val = hist.history['val_loss']
    epochs = range(1,41)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    accuracy_train = hist.history['accuracy']
    accuracy_val = hist.history['val_accuracy']
    epochs = range(1,41)
    plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
    plt.plot(epochs, accuracy_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def PrintConfusionMatrix(y_test,y_pred,label):
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(14,14))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=label, yticklabels=label ,cmap=plt.cm.Blues)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    fig.savefig(path+'model_result/'+'ConfusionMatrix'+m+str(consider_unclassified)+'.png', dpi=600)

def GetFilePDB():
    inp = files.upload()
    dataSet = pd.DataFrame(columns=['pdb_id','s_ch','s_resi','s_ins','s_resn','s_ss8','s_rsa','s_up','s_down','s_phi','s_psi','s_ss3','s_a1','s_a2','s_a3', 's_a4', 's_a5', 't_ch', 't_resi','t_ins','t_resn','t_ss8','t_rsa','t_up',	't_down',	't_phi',	't_psi',	't_ss3',	't_a1',	't_a2',	't_a3',	't_a4',	't_a5',	'Interaction'])
    for key in inp.keys():
      data = pd.read_csv(key, sep='\t')
      dataSet = dataSet.append(data, ignore_index=True)

    dataSet.to_csv(path+'SB_dataset.csv')
    return dataSet

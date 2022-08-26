import numpy as np
import csv
import wordninja as wn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def arrayToSentence(x):
    string=''
    for a in x:
        string=string + a + ' '
    return string

def buildMulti():

    filecsv= open("/content/drive/MyDrive/Cyber Security/Elmo/Dataset_Completo.csv", newline="")
    lettore = csv.reader(filecsv, delimiter=";")

    sentencesTemp=[]
    labelsTemp=[]

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    labelTemp=''

    for a in lettore:

        if labelTemp=='':
            labelTemp=a[1]
            splitted = wn.split(a[3])
            sentence = arrayToSentence(splitted)
            sentencesTemp.append(sentence)
            labelsTemp.append(a[1])
        elif labelTemp==a[1]:
            splitted = wn.split(a[3])
            sentence = arrayToSentence(splitted)
            sentencesTemp.append(sentence)
            labelsTemp.append(a[1])
        elif a[1] != labelTemp:
            x_trainTemp, x_testTemp, y_trainTemp, y_testTemp = train_test_split(sentencesTemp,labelsTemp,test_size=0.2)
            x_train=x_train + x_trainTemp
            x_test=x_test + x_testTemp
            y_train=y_train + y_trainTemp
            y_test=y_test + y_testTemp
            sentencesTemp=[]
            labelsTemp=[]
            labelTemp=a[1]
            splitted = wn.split(a[3])
            sentence = arrayToSentence(splitted)
            sentencesTemp.append(sentence)
            labelsTemp.append(a[1])

    filecsv.close()

    x_trainTemp, x_testTemp, y_trainTemp, y_testTemp = train_test_split(sentencesTemp,labelsTemp,test_size=0.2)
    x_train=x_train + x_trainTemp
    x_test=x_test + x_testTemp
    y_train=y_train + y_trainTemp
    y_test=y_test + y_testTemp

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)


    return x_train, x_test, y_train, y_test

def buildBinary():

    filecsv= open("/content/drive/MyDrive/Cyber Security/Elmo/Dataset_Completo.csv", newline="")
    lettore = csv.reader(filecsv, delimiter=";")

    sentencesTemp=[]
    labelsTemp=[]

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    labelTemp=''

    for a in lettore:

        if labelTemp=='':
            labelTemp=a[0]
            splitted = wn.split(a[3])
            sentence = arrayToSentence(splitted)
            sentencesTemp.append(sentence)
            labelsTemp.append(a[0])
        elif labelTemp==a[0]:
            splitted = wn.split(a[3])
            sentence = arrayToSentence(splitted)
            sentencesTemp.append(sentence)
            labelsTemp.append(a[0])
        elif a[0] != labelTemp:
            x_trainTemp, x_testTemp, y_trainTemp, y_testTemp = train_test_split(sentencesTemp,labelsTemp,test_size=0.20)
            x_train=x_train + x_trainTemp
            x_test=x_test + x_testTemp
            y_train=y_train + y_trainTemp
            y_test=y_test + y_testTemp
            sentencesTemp=[]
            labelsTemp=[]
            labelTemp=a[0]
            splitted = wn.split(a[3])
            sentence = arrayToSentence(splitted)
            sentencesTemp.append(sentence)
            labelsTemp.append(a[0])

    filecsv.close()

    x_trainTemp, x_testTemp, y_trainTemp, y_testTemp = train_test_split(sentencesTemp,labelsTemp,test_size=0.20)
    x_train=x_train + x_trainTemp
    x_test=x_test + x_testTemp
    y_train=y_train + y_trainTemp
    y_test=y_test + y_testTemp

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)


    return x_train, x_test, y_train, y_test

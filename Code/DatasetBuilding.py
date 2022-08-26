import numpy as np

import wordninja as wn
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def arrayToSentence(x):
    string=''
    for a in x:
        string=string + a + ' '
    return string

def build():

    numOsservazioni = 960

    BADsplittedlist = []
    TOPsplittedlist = []
    GOODsplittedlist = []

    sentencesBAD=[]
    sentencesGOOD=[]
    labelsBAD=[]
    labelsGOOD=[]

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    file = open('/content/drive/MyDrive/Cyber Security/Elmo/matsnu.txt', 'r' )
    for n in range(numOsservazioni):
        element = wn.split(file.readline())
        elementString = arrayToSentence(element)
        BADsplittedlist.append(elementString)
        labelsBAD.append('DGA')
    file.close()


    file1 = open('/content/drive/MyDrive/Cyber Security/Elmo/opendns-top-domains.txt', 'r')
    for n in range(10000):
        element = wn.split(file1.readline())
        elementString = arrayToSentence(element)
        TOPsplittedlist.append(elementString)
        labelsGOOD.append('legit')
    file1.close()

    file2 = open('/content/drive/MyDrive/Cyber Security/Elmo/opendns-random-domains.txt', 'r')
    for n in range(7500):
        element = wn.split(file2.readline())
        elementString = arrayToSentence(element)
        GOODsplittedlist.append(elementString)
        labelsGOOD.append('legit')
    file2.close()

    sentencesBAD=BADsplittedlist
    sentencesGOOD=TOPsplittedlist+GOODsplittedlist

    x_trainBAD, x_testBAD, y_trainBAD, y_testBAD = train_test_split(sentencesBAD,labelsBAD,test_size=0.2)
    x_train=x_train + x_trainBAD
    x_test=x_test + x_testBAD
    y_train=y_train + y_trainBAD
    y_test=y_test + y_testBAD

    x_trainGOOD, x_testGOOD, y_trainGOOD, y_testGOOD = train_test_split(sentencesGOOD,labelsGOOD,test_size=0.2)
    x_train=x_train + x_trainGOOD
    x_test=x_test + x_testGOOD
    y_train=y_train + y_trainGOOD
    y_test=y_test + y_testGOOD

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)


    return x_train, x_test, y_train, y_test

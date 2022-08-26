import numpy as np

import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Lambda,Input

import tensorflow_hub as hub
import tensorflow as tf

import DatasetBuilding as DsB
import DatasetBuildingMorbidoni as DsBM

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


batch_size = 32
numEpochs = 1 #10

print('Dataset Building...')

#x_train, x_test, y, y_test=DsB.build()
#Type = 'paper-matsnu-binary-'

x_train, x_test, y, y_test=DsBM.buildBinary()
Type = 'binary-'

#x_train, x_test, y, y_test=DsBM.buildMulti()
#Type = 'multiclass-'


x_train=np.array(x_train)
x_test=np.array(x_test)


le = preprocessing.LabelEncoder()
le.fit(y)

# Definisco le funzioni per codificare e decodificare le labels
def encode(le, labels):
    enc = le.transform(labels)
    return tf.keras.utils.to_categorical(enc)  #era solo keras.utils...

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

y_enc = encode(le, y)

y_train=np.array(y_enc)

print('...done!')

print('Model Construction...')
#Parte costruzione del modello
# importo il modulo con la funzione di embedding ELMo
elmo = hub.Module("/content/drive/MyDrive/Cyber Security/Elmo/3")

# Definisco la funzione di embedding
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense=Dense(128, activation='relu')(embedding)
pred=Dense(len(y_enc[0]), activation='sigmoid')(dense)
model=Model(inputs=[input_text], outputs=pred)


model.compile('adam', 'binary_crossentropy', metrics=['accuracy',
    tf.keras.metrics.AUC(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.Accuracy(), #aggiunto questo
    ])

print('...done!')

print('Train...')
#parte di training
with tf.compat.v1.Session() as session:
    tf.compat.v1.keras.backend.set_session(session)
    session.run(tf.compat.v1.global_variables_initializer())
    session.run(tf.compat.v1.tables_initializer())
    history = model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=numEpochs,
                #validation_data=[x_val, y_val]
              )
    model.save_weights('/content/drive/MyDrive/Cyber Security/Elmo/Saved_weights/' + Type + 'elmo-model.h5')

    print('\nhistory dict:', history.history)

print('...done!')

print('Test...')
#Parte di testing
with tf.compat.v1.Session() as session:
    tf.compat.v1.keras.backend.set_session(session)
    session.run(tf.compat.v1.global_variables_initializer())
    session.run(tf.compat.v1.tables_initializer())
    model.load_weights('/content/drive/MyDrive/Cyber Security/Elmo/Saved_weights/'+ Type + 'elmo-model.h5')
    predicts = model.predict(x_test, batch_size=32)

y_preds = decode(le, predicts)

print('...done!')

print('Results:')
#Plotta i risultati
cm = metrics.confusion_matrix(y_test, y_preds)
print(metrics.classification_report(y_test, y_preds))

df_cm = pd.DataFrame(cm, index = [i for i in le.classes_],
                  columns = [i for i in le.classes_])
plt.figure(1, figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt="d")
plt.show()

#fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test.ravel(), predicts.ravel())
#auc_keras = auc(fpr_keras, tpr_keras)

#plotta i grafici
#plt.figure(2, figsize = (10,7))
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_keras, tpr_keras, label='Binary (area = {:.3f})'.format(auc_keras))
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best')
#plt.show()

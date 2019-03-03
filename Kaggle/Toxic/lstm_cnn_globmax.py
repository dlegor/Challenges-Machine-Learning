import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import optimizers

print("Definition Object")
print("="*100)
Input=layers.Input
Dense=layers.Dense
Dropout=layers.Dropout
Embedding=layers.Embedding
SpatialDropout1D=layers.SpatialDropout1D
concatenate=layers.concatenate
Model=models.Model

LSTM=layers.LSTM
Conv1D=layers.Conv1D
Bidirectional=layers.Bidirectional
GlobalAveragePooling1D=layers.GlobalAveragePooling1D
GlobalMaxPooling1D=layers.GlobalMaxPooling1D
Adam=optimizers.Adam


text=preprocessing.text
sequence=preprocessing.sequence

Callback=callbacks.Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'


print("Load Data")
EMBEDDING_FILE = '/home/dlegorreta/Documentos/Kaggle/Toxic_Comments/Extra_data/Mod_300_2W.vec'

train = pd.read_csv('/home/dlegorreta/Documentos/Kaggle/Toxic_Comments/train.csv.zip',compression='zip',nrows=50000)
test = pd.read_csv('/home/dlegorreta/Documentos/Kaggle/Toxic_Comments/test.csv.zip',compression='zip',nrows=50000)
submission = pd.read_csv('/home/dlegorreta/Documentos/Kaggle/Toxic_Comments/sample_submission.csv.zip',compression='zip')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

print("Definition Model")
print("="*100)
def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid")(x)
    #avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    dx = Dropout(0.5)(max_pool)
    #conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(dx)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=3e-3),
                  metrics=['accuracy'])

    return model


model = get_model()


batch_size = 32
epochs = 10

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

print("Estimation Model")
print("="*100)
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)


y_pred = model.predict(x_test, batch_size=1024)


print("Make Prediction")
print("="*100)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred

print("finishe Estimation")
print("="*100)
submission.to_csv('/home/dlegorreta/Documentos/Kaggle/Toxic_Comments/submission_LSTM128_CNN64_GlobMax.csv', index=False)

print("Save Model")
print("="*100)
 
model.save("nn_3_LSTM128_CNN64_GlobalMax.h5")

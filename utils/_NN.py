import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

np.random.seed(1234)

feat_df = pd.read_pickle('nn_feat_train.pkl')

feat_df = shuffle(feat_df)
X = np.rollaxis(np.dstack(feat_df['seq_scaled'].values), -1)
X_train, y_train = X, feat_df['flop']

callbacks = [(ReduceLROnPlateau(monitor='val_accuracy', factor=.25, patience=1, verbose=1, mode='max')),
             (EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, mode='max')),
             (ModelCheckpoint('weights_gru.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='max'))]

model = Sequential()
model.add(GRU(100, input_shape=(365, 6), return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(50, input_shape=(365, 6), return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(20, input_shape=(365, 6), return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(10, input_shape=(365, 6), return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(5, input_shape=(365, 6), return_sequences=True))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(learning_rate=0.00001)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=100, validation_split=.25, batch_size=128,
          callbacks=callbacks)


model_json = model.to_json()
with open("model_gru.json", "w") as json_file:
    json_file.write(model_json)

callbacks = [(ReduceLROnPlateau(monitor='val_accuracy', factor=.25, patience=1, verbose=1, mode='max')),
             (EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, mode='max')),
             (ModelCheckpoint('weights_lstm.h5', monitor='val_accuracy', save_best_only=True,
                              save_weights_only=False, mode='max'))]

model = Sequential()
model.add(LSTM(100, input_shape=(365, 6), return_sequences=True))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=100, validation_split=.25, batch_size=128,
          callbacks=callbacks)


model_json = model.to_json()
with open("model_lstm.json", "w") as json_file:
    json_file.write(model_json)
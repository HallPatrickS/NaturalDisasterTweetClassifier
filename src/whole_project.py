'''
Neraul networking thing
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data import load_disaster_tweets, verify_labels, ngram_vectorize

import params
import model as m


(X, Y) = load_disaster_tweets(params.DATA_PATH, 123)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

verify_labels(Y_test)
X_vect = ngram_vectorize(X, Y)

units, activation = m._get_last_layer_units_and_activation(4)

model = m.mlp_model(params.NUM_LAYERS, units, params.DROPOUT, X_vect.shape, params.NUM_CLASSES)
loss = m._get_loss()

optimizer = Adam(lr=params.LEARNING_RATE)
model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

callbacks = [EarlyStopping(monitor='loss', patience=2)]

print(model.summary())
print(str(model))

Y_np = np.array(Y)[np.newaxis, :, np.newaxis]

hist = model.fit(
        X_vect[np.newaxis, :, :],
        Y_np,
        epochs=params.EPOCHS,
        callbacks=callbacks,
        # validation_data=(X_test, np.array(Y_test)),
        # validation_split=0.2,
        verbose=1,  # Logs once per epoch.
        batch_size=2,
)

from keras.utils import plot_model
plot_model(model, to_file='model.png')

history = hist.history 
print(hist.history)
print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['acc'][-1], loss=history['loss'][-1]))

# Save model.
model.save('IMDb_mlp_model.h5')
# return history['val_acc'][-1], history['val_loss'][-1]
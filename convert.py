import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

import joblib

import tensorflow as tf

import numpy as np

# define our model, note: we can't just load a saved model as we trained it
# with GPU versions of the LSTM, and need to use the non-GPU ones here
model = Sequential()
model.add(Embedding(472607, 100, input_length=56, trainable=False))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load the weights from our training run
model.load_weights("tweet_model_weights.h5")

# Extract the embedding matrix from the first layer
embedding_matrix = model.layers[0].get_weights()[0]
print( 'Embedding Shape ~> {}'.format( embedding_matrix.shape ) )

# Save the embedding matrix for a compressed pickle
joblib.dump(embedding_matrix, "embedding_matrix.pkl.bz2")

# Create some dummy input
input = np.zeros((1, 56, 100))

# Create a new model, removing the embedding layer
new_model = Sequential(model.layers[1:])
new_model.build(input_shape=input.shape)

# We need to do a prediction to get TF to setup the right shape
new_model.predict(input)

# Convert the model to tflite, enabling quantization to make
# it a bit smaller
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
converter.post_training_quantize = True
tflite_buffer = converter.convert()

# Save the tflite file out
with open( "tweet_model.tflite" , "wb" ) as f:
    f.write( tflite_buffer )

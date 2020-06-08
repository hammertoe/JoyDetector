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

#tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

model = Sequential()
model.add(Embedding(472607, 100, input_length=56, trainable=False))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
# compile the model                                                                                                                                                                                                                   
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("tweet_model_weights.h5")


embedding_matrix = model.layers[0].get_weights()[0]
print( 'Embedding Shape ~> {}'.format( embedding_matrix.shape ) )

#word_index = tokenizer.word_index
#word_index_2 = dict()
#for word , index in word_index.items():
#    word_index_2[ index ] = word
#word_index = word_index_2
#embedding_dict = dict()


#for i in range( len( embedding_matrix ) - 1 ):
#    embedding_dict[ word_index[ i + 1 ] ] = embedding_matrix[ i + 1 ]


joblib.dump(embedding_matrix, "embedding_matrix.pkl.bz2")
#with open("embedding_matrix.pkl", "wb") as f:
#    pickle.dump(embedding_matrix, f)

input = np.zeros((1, 56, 100))

new_model = Sequential(model.layers[1:])
new_model.build(input_shape=input.shape)
#new_model.save("noembed_tweet_model.h5")

# We need to do a prediction to get TF to setup the right shape
new_model.predict(input)

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
converter.post_training_quantize = True
tflite_buffer = converter.convert()
with open( "tweet_model.tflite" , "wb" ) as f:
    f.write( tflite_buffer )

#model.save("tweet_model.h5")

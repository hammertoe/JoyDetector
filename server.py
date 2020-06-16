import numpy as np
import os

import joblib
from urllib.request import urlopen

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import tflite_runtime.interpreter as tflite

from flask import Flask, request

app = Flask(__name__)

tokenizer_url = os.getenv('TOKENIZER_URL',
                          f'file://{os.getcwd()}/tokenizer.pkl.bz2')
print("Loading tokenizer from:", tokenizer_url)
tokenizer = joblib.load(urlopen(tokenizer_url))
embedding_matrix_url = os.getenv('EMBEDDING_MATRIX_URL',
                                 f'file://{os.getcwd()}/embedding_matrix.pkl.bz2')
print("Loading embedding matrix from:", embedding_matrix_url)
embedding_matrix = joblib.load(urlopen(embedding_matrix_url))
print("loaded")

# Load the TFLite model and allocate tensors.
global interpreter
interpreter = tflite.Interpreter(model_path="tweet_model.tflite")

# Allocate the tensor
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/tones', methods=['POST'])
def get_tones():
    texts = request.json['texts']
    tkns = tokenizer.texts_to_sequences(texts)
    padded_docs = pad_sequences(tkns, maxlen=56, padding='post')

    def process():
        for doc in padded_docs:
            input = np.zeros((1, padded_docs.shape[1], 100), dtype=np.float32)
            for i, x in enumerate(doc):
                input[0, i] = embedding_matrix[x]

            # Set the input
            interpreter.set_tensor(input_details[0]['index'], input)

            # Run the interpreter
            interpreter.invoke()

            # Get the outputs
            preds = interpreter.get_tensor(output_details[0]['index'])

            yield preds[0]

    res = [{'text': texts[i], 'joy':round(float(x[0]), 4), 'anger':round(float(x[1]), 4)} for i, x in enumerate(tuple(process()))]
    return {'tones': res}


port = int(os.getenv('PORT', 8000))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)

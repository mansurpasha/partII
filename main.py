from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Dense, LSTM
from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time
import argparse

print(tf.__version__)

#own modules

import models
import processing
import args
import inference

parser = argparse.ArgumentParser()
args.format_parser(parser)

parameters, unparsed = parser.parse_known_args()

path_to_file = parameters.train_file
print(path_to_file)


#######DATA PREPROCESSING#######

# Try experimenting with the size of that dataset
num_examples = 324401
input_tensor, target_tensor_in, target_tensor_out, language, max_length = processing.load_dataset(path_to_file, num_examples)

# Creating training and validation sets using an 80-20 split
#input_tensor_train, input_tensor_val, target_tensor_in_train, target_tensor_in_val, target_tensor_out_train, target_tensor_out_val = train_test_split(input_tensor, target_tensor_in, target_tensor_out, test_size=0.2)

# Show length
#print(len(input_tensor_train), len(target_tensor_in_train), len(target_tensor_out_train), len(input_tensor_val), len(target_tensor_in_val), len(target_tensor_out_val))

#BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
#N_BATCH = BUFFER_SIZE//BATCH_SIZE
EPOCHS = 10
embedding_dim = 128
units = 256
vocab_size = len(language.word2idx)

encoder_input_data = input_tensor
decoder_input_data = target_tensor_in
decoder_output_data = target_tensor_out

#######TRAINING#######

if (parameters.model_file == None):
    #define model structure for encoder of training model
    encoder_input = Input(shape=(None,), name="Encoder_input")
    encoder = LSTM(units, return_state=True, name='Encoder_lstm')
    #use same word embeddings in both encoding and decoding phase
    Embedding_layer = Embedding(output_dim=embedding_dim, input_dim=vocab_size, name="Embedding")
    word_embedding_context = Embedding_layer(encoder_input)
    encoder_output, state_h, state_c = encoder(word_embedding_context)
    encoder_states = [state_h, state_c]
    #define model structure for decoder of training model
    decoder_input = Input(shape=(None,), name="Decoder_input")
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="Decoder_lstm")
    word_embedding_answer = Embedding_layer(decoder_input)
    decoder_output, _, _ = decoder_lstm(word_embedding_answer, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name="Dense_layer")
    decoder_output = decoder_dense(decoder_output)
    #compile model
    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    encoder_model = Model(encoder_input, encoder_states)
    decoder_state_input_h = Input(shape=(units,), name="H_state_input")
    decoder_state_input_c = Input(shape=(units,), name="C_state_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    encoder_model.summary()
    decoder_model.summary()
else:
    model = load_model(parameters.model_file)

model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

model.save("seq2seq_model")

encoder, decoder = inference.build_inference_models(units, embedding_dim, vocab_size, model.get_weights())

for k in range(10):
    print(inference.decode_sequence(input_tensor[k:k+1],encoder,decoder,language,max_length))
    print(inference.decode_sequence(input_tensor[k:k+1],encoder_model,decoder_model,language,max_length))



from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Dense, LSTM, GRU
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

#encoder_input_data = tf.zeros(len(input_tensor_train),)
#dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#######BUILD MODEL#######
'''
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, embedding_dim, input_length=max_length_inp)(encoder_inputs)
x, state_h = GRU(latent_dim, return_state=True)(x)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, embedding_dim, input_length=max_length_targ)(decoder_inputs)
print(x.shape)
x = GRU(latent_dim, return_sequences=True)(x, initial_state=state_h)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)
'''
#######TRAINING#######
'''
checkpoint_dir = '/home/map79/git/partII/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


encoder_inputs = Input(shape=(max_length,), name="Encoder_input", )
encoder = GRU(units, return_state=True, name='Encoder_lstm')
Shared_Embedding = Embedding(output_dim=embedding_dim, input_dim=vocab_size, name="Embedding")
word_embedding_context = Shared_Embedding(encoder_inputs)
encoder_outputs, state_h = encoder(word_embedding_context)
decoder_inputs = Input(shape=(max_length,), name="Decoder_input")
decoder_lstm = GRU(units, return_sequences=True, return_state=True, name="Decoder_lstm")
word_embedding_answer = Shared_Embedding(decoder_inputs)
decoder_outputs, _ = decoder_lstm(word_embedding_answer, initial_state=state_h)
decoder_dense = Dense(vocab_size, activation='softmax', name="Dense_layer")
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

encoder_model = Model(encoder_inputs, state_h)
decoder_state_input_h = Input(shape=(units,), name="H_state_input")
decoder_state_input = decoder_state_input_h
decoder_outputs, state_h = decoder_lstm(word_embedding_answer, initial_state=decoder_state_input)
decoder_state = state_h
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + [decoder_state_input], [decoder_outputs] + [decoder_state])

encoder_model.summary()
decoder_model.summary()
'''
if (parameters.model_file == None):
    encoder_inputs = Input(shape=(None,), name="Encoder_input")
    encoder = LSTM(units, return_state=True, name='Encoder_lstm')
    Shared_Embedding = Embedding(output_dim=embedding_dim, input_dim=vocab_size, name="Embedding")
    word_embedding_context = Shared_Embedding(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(word_embedding_context)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None,), name="Decoder_input")
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="Decoder_lstm")
    word_embedding_answer = Shared_Embedding(decoder_inputs)
    decoder_outputs, _, _ = decoder_lstm(word_embedding_answer, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name="Dense_layer")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    module.compile(optimizer='rmsprop', loss='categorical_crossentropy')
else:
    model = load_model(parameters.model_file)
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

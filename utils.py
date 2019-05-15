import processing
import models
from nltk.tokenize import word_tokenize

from __future__ import absolute_import, division, print_function

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time
import argparse
import random
import pickle

from processing import load_preprocess
import models
import args

def sentence_to_idx_padded(sentence, lang, max_len):
    indices = [lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in word_tokenize(sentence.lower())]
    padded = tf.keras.preprocessing.sequence.pad_sequences([indices],
                                                           maxlen=max_len,
                                                           padding='post')
    return padded

def sentence_to_idx(sentence, lang):
    indices = [lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in word_tokenize(sentence.lower())]
    return np.array(indices)

def idx_to_sentence(sentence, lang):
    output = ""
    for word in sentence:
        if lang.idx2word[word] != "<pad>":
            output += ' ' + lang.idx2word[word]
    return output[1:]


def calculate_probability(encoder, decoder, sentence, expected_response, lang):
    inputs = tf.convert_to_tensor(sentence)
    inputs = tf.expand_dims(inputs, 0)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([lang.word2idx['<start>']], 0)

    running_probability = 1

    expected_response = sentence_to_idx_unpadded(expected_response, lang)
    expected_response = expected_response.tolist()
    expected_response.append(lang.word2idx["<end>"])
    expected_response = np.array(expected_response)

    for word in expected_response:
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        probability = predictions[0][word].numpy()

        running_probability *= probability

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([word], 0)

    return running_probability

def find_common_responses(sentences):
    return


out = calculate_probability(encoder, decoder,
                            input_tensor_train[0],
                            "I don't know",
                            language)
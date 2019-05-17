import processing
import models
from nltk.tokenize import word_tokenize

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


def load_preprocess(datafile):
    with open(datafile, mode='rb') as in_file:
        return pickle.load(in_file)



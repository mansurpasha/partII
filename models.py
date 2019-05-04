from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Embedding

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

print(tf.__version__)

def process_decoder_input(target_data, language, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = language.word2idx['<start>']

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                               target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, language, max_target_sequence_length,
                         vocab_size, output_layer, keep_prob, batch_size):
    """
    Create a inference process in decoding layer
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                      tf.fill([batch_size], language.word2idx["<start>"]),
                                                      language.word2idx["<end>"])

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


def seq2seq_model(input, target, target_length, max_target_length, params, lang_dict):

    print(target.shape)

    vocab_size = len(lang_dict.word2idx)
    # Embedding layer
    embedding = tf.get_variable("embeddings", [vocab_size, params.embedding_dim])
    embedded_enc_inp = tf.nn.embedding_lookup(embedding, input)

    # Construct encoder RNN
    encoder_cell = tf.nn.rnn_cell.LSTMCell(params.units, name="encoder_lstm")
    enc_outputs, enc_state = tf.nn.dynamic_rnn(encoder_cell,
                                       embedded_enc_inp,
                                       dtype=tf.float32)

    # Prepare data for decoder, reuse same embedding layer
    dec_input = process_decoder_input(target, lang_dict, params.batch_size)
    print(dec_input.shape)
    dec_embed_input = tf.nn.embedding_lookup(embedding, dec_input)
    print(dec_embed_input.shape)

    decoder_cell = tf.nn.rnn_cell.LSTMCell(params.units, name="decoder_lstm")

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(vocab_size)
        train_output = decoding_layer_train(enc_state,
                                            decoder_cell,
                                            dec_embed_input,
                                            target_length,
                                            max_target_length,
                                            output_layer,
                                            params.keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(enc_state,
                                            decoder_cell,
                                            dec_embed_input,
                                            lang_dict,
                                            max_target_length,
                                            vocab_size,
                                            output_layer,
                                            params.keep_prob,
                                            params.batch_size)

    return train_output, infer_output

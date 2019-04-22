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

def process_decoder_input(target_data, language, params.batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = language.word2idx['<start>']

    after_slice = tf.strided_slice(target_data, [0, 0], [params.batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([params.batch_size, 1], go_id), after_slice], 1)

    return after_concat

def encoding_layer(rnn_inputs, params.units, params.num_layers, params.keep_prob,
                   vocab_size,
                   params.embedding_dim):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs,
                                             vocab_size=vocab_size,
                                             embed_dim=params.embedding_dim)

    stacked_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params.units), params.keep_prob) for _ in range(params.num_layers)])

    outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                       embed,
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, params.keep_prob):
    """
    Create a training process in decoding layer
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=params.keep_prob)

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
                         vocab_size, output_layer, params.batch_size, params.keep_prob):
    """
    Create a inference process in decoding layer
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=params.keep_prob)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                      tf.fill([params.batch_size], language.word2idx["<start>"]),
                                                      language.word2idx["<end>"])

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   params.units,
                   params.num_layers, language, vocab_size,
                   params.batch_size, params.keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    vocab_size = len(language.word2idx)
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(params.units) for _ in range(params.num_layers)])

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(vocab_size)
        train_output = decoding_layer_train(encoder_state,
                                            cells,
                                            dec_embed_input,
                                            target_sequence_length,
                                            max_target_sequence_length,
                                            output_layer,
                                            params.keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state,
                                            cells,
                                            dec_embeddings,
                                            language,
                                            max_target_sequence_length,
                                            vocab_size,
                                            output_layer,
                                            params.batch_size,
                                            params.keep_prob)

    return (train_output, infer_output)


def seq2seq_model(input_data, target_data, params.keep_prob, params.batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  vocab_size,
                  params.embedding_dim, params.embedding_dim,
                  params.units, params.num_layers, language):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data,
                                             params.units,
                                             params.num_layers,
                                             params.keep_prob,
                                             vocab_size,
                                             params.embedding_dim)

    dec_input = process_decoder_input(target_data,
                                      language,
                                      params.batch_size)

    train_output, infer_output = decoding_layer(dec_input,
                                                enc_states,
                                                target_sequence_length,
                                                max_target_sentence_length,
                                                params.units,
                                                params.num_layers,
                                                language,
                                                vocab_size,
                                                params.batch_size,
                                                params.keep_prob,
                                                params.embedding_dim)

    return train_output, infer_output

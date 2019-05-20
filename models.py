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

def process_decoder_input(target_data, go_id, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat

# calculates the average of a tensor
def reduce_mean(tensor):
    total = sum(tensor)
    return tf.convert_to_tensor(total/len(tensor))

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

    vocab_size = len(lang_dict.word2idx)
    # Embedding layer
    embedding = tf.get_variable("embeddings", [vocab_size, params.embedding_dim])
    embedded_enc_inp = tf.nn.embedding_lookup(embedding, input)

    # Construct encoder RNN
    stacked_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params.units), params.keep_prob) for _ in
         range(params.num_layers)])

    encoder_cell = tf.contrib.rnn.LSTMCell(params.units, name="encoder_lstm")
    enc_outputs, enc_state = tf.nn.dynamic_rnn(encoder_cell,
                                       embedded_enc_inp,
                                       dtype=tf.float32)

    # Prepare data for decoder, reuse same embedding layer
    dec_input = process_decoder_input(target, lang_dict, params.batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embedding, dec_input)

    decoder_cell = tf.contrib.rnn.LSTMCell(params.units)

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
                                            embedding,
                                            lang_dict,
                                            max_target_length,
                                            vocab_size,
                                            output_layer,
                                            params.keep_prob,
                                            params.batch_size)





    return train_output, infer_output

# Embedding layer converts (batch x seq_len x 1) to (batch x seq_len x 1 x embedding_dim)
# This function removes the 1 from a tensor with an unspecified shape (necessary due to use of placeholders)
def reshape_embeddings(embedded_seq, embedding_dim):
     return tf.reshape(embedded_seq, [tf.shape(embedded_seq)[0], tf.shape(embedded_seq)[1], embedding_dim])

'''
class RLModel:
    def __init__(self, params, language, name="RL_Model"):
        self.params = params
        self.language = language
        self.vocab_size = len(language.word2idx)

        with tf.variable_scope(name):
            # Model inputs for pretraining the Seq2Seq model
            with tf.name_scope("inputs"):
                # Input of size (batch_size * max_seq_len {of batch} * num_features {single word id})
                self.inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs_")
                self.inputs_lengths_ = tf.placeholder(tf.int32, [None], name="input_lengths_")
            with tf.name_scope("S2S_train_inputs"):
                self.targets_ = tf.placeholder(tf.int32, [None, None], name="targets_")
                self.target_lengths_ = tf.placeholder(tf.int32, [None], name="target_lengths_")
                self.max_target_length_ = tf.placeholder(tf.int32, shape=(), name="target_max_length_")

            # Inputs required for reinforcement learning
            with tf.name_scope("RL_inputs"):
                # List of actions taken from episode batches
                self.actions_ = tf.placeholder(tf.int32, [None], name="actions_")
                # List of discounted rewards calculated from episode batches
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
                # Placeholder variable to allow tensorboard tracking
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            # Embed inputs, embedding layer reused when decoding
            with tf.name_scope("embedding"):
                self.embeddings = tf.get_variable("embeddings", [self.vocab_size, self.params.embedding_dim])
                self.embedded_enc_inp = tf.nn.embedding_lookup(self.embeddings, self.inputs_)
                #self.embedded_enc_inp = reshape_embeddings(self.embedded_enc_inp, self.params.embedding_dim)
            with tf.name_scope("encoding"):
                # Construct encoder RNN, num_layers deep, with dropout
                self.encoder_cell = stack_RNN_cells(params, None)
                # self.encoder_cell = tf.nn.rnn_cell.LSTMCell(params.units, name="encoder_lstm")
                self.enc_outputs, self.enc_state = tf.nn.dynamic_rnn(self.encoder_cell,
                                                                     self.embedded_enc_inp,
                                                                     dtype=tf.float32)

            
            with tf.name_scope("attention"):
                # Construct attention mechanism using outputs from encoder
                # attention_states: [batch_size, max_time, num_units]
                self.attention_states = tf.transpose(self.enc_outputs, [1, 0, 2])

                # Create an attention mechanism
                self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.params.units, self.params.attention_state,
                    memory_sequence_length=self.inputs_lengths_)
            

            with tf.name_scope("decoding"):
                # Prepare data for decoder, reuse same embedding layer

                self.dec_embed_input = tf.nn.embedding_lookup(self.embeddings, self.targets_)

                self.decoder_cell = stack_RNN_cells(params, self.attention_mechanism)

                self.output_layer = tf.layers.Dense(self.vocab_size)

                # this part of the graph is used for training batches
                with tf.name_scope("training"):
                    self.helper = tf.contrib.seq2seq.TrainingHelper(self.dec_embed_input, self.target_lengths_)
                    self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                                   self.helper,
                                                                   self.enc_state,
                                                                   self.output_layer)

                    # unrolling the decoder layer
                    self.training_output, _, _ = tf.contrib.seq2seq.dynamic_decode(self.decoder,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=self.max_target_length_)

                # use this part of the graph for inferring unseen sentences
                with tf.variable_scope("inference"):
                    self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                                           tf.fill([self.params.batch_size],
                                                                                   language.word2idx["<start>"]),
                                                                           language.word2idx["<end>"])

                    if params.beam_search == 'n':
                        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                                       self.helper,
                                                                       self.enc_state,
                                                                       self.output_layer)
                    else:
                        # make beam width copies of the initial state to start the search
                        self.decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                            self.enc_state, multiplier=self.params.beam_width)

                        # initialize beam decoder
                        self.decoder = tf.contrib.seq2seq.BeamDecoder(self.decoder_cell,
                                                                      self.helper,
                                                                      tf.fill([self.params.batch_size],
                                                                              language.word2idx["<start>"]),
                                                                      self.language.word2idx["<end>"],
                                                                      self.decoder_initial_state,
                                                                      beam_width=params.beam_width,
                                                                      output_layer=self.output_layer,
                                                                      length_penalty_weight=0,
                                                                      coverage_penalty_weight=0)

                    # unrolling the decoder layer
                    self.inference_output, _, _  = tf.contrib.seq2seq.dynamic_decode(self.decoder,
                                                                                     impute_finished=True,
                                                                                     maximum_iterations=self.max_target_length_)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inference_output[0], labels=self.actions_)
                self.loss = reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("train"):
                # call this operation to train the model using the loss value
                self.train_op = tf.train.RMSPropOptimizer(self.params.learning_rate).minimize(self.loss)
'''

class RLModel:
    # Modifies basic LSTM cell with additional wrappers as necessary, defined by params
    def create_RNN_cell(self,params):
        if params.use_dropout:
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params.units), params.keep_prob)
        else:
            return tf.contrib.rnn.LSTMCell(params.units)

    # Takes single RNN parameters and stacks them on top of each other in a MultiRNNCell if num_layers is greater than 1
    def stack_RNN_cells(self,params):
        if params.num_layers > 1:
            return tf.contrib.rnn.MultiRNNCell([self.create_RNN_cell(params) for _ in range(params.num_layers)])
        else:
            return self.create_RNN_cell(params)

    # calculates the average of a tensor
    def reduce_mean(self, tensor):
        total = sum(tensor)
        return tf.convert_to_tensor(total / len(tensor))

    def __init__(self, params, language, name="RL_Model"):
        self.params = params
        self.language = language
        self.vocab_size = len(language.word2idx)

        with tf.variable_scope(name):
            # Model inputs for pretraining the Seq2Seq model
            with tf.name_scope("inputs"):
                # Input of size (batch_size * max_seq_len {of batch} * num_features {single word id})
                self.inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs_")
                self.input_lengths_ = tf.placeholder(tf.int32, [None], name="input_lengths_")
            with tf.name_scope("S2S_train_inputs"):
                self.targets_ = tf.placeholder(tf.int32, [None, None], name="targets_")
                self.target_lengths_ = tf.placeholder(tf.int32, [None], name="target_lengths_")
                self.max_target_length_ = tf.placeholder(tf.int32, shape=(), name="target_max_length_")

            # Inputs required for reinforcement learning
            with tf.name_scope("RL_inputs"):
                # List of actions taken from episode batches
                self.actions_ = tf.placeholder(tf.int32, [None, 10004], name="actions_")
                # List of discounted rewards calculated from episode batches
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
                # Placeholder variable to allow tensorboard tracking
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            # Embed inputs, embedding layer reused when decoding
            with tf.name_scope("embedding"):
                self.embeddings = tf.get_variable("embeddings", [self.vocab_size, self.params.embedding_dim])
                self.embedded_enc_inp = tf.nn.embedding_lookup(self.embeddings, self.inputs_)
                #self.embedded_enc_inp = reshape_embeddings(self.embedded_enc_inp, self.params.embedding_dim)
            with tf.name_scope("encoding"):
                # Construct encoder RNN, num_layers deep, with dropout
                self.encoder_cell = self.stack_RNN_cells(params)
                # self.encoder_cell = tf.nn.rnn_cell.LSTMCell(params.units, name="encoder_lstm")
                self.enc_outputs, self.enc_state = tf.nn.dynamic_rnn(self.encoder_cell,
                                                                     self.embedded_enc_inp,
                                                                     dtype=tf.float32)

            with tf.name_scope("decoding"):
                # Prepare data for decoder, reuse same embedding layer

                self.dec_embed_input = tf.nn.embedding_lookup(self.embeddings, self.targets_)

                self.decoder_cell = self.stack_RNN_cells(params)

                self.output_layer = tf.layers.Dense(self.vocab_size)

                # this part of the graph is used for training batches
                with tf.name_scope("training"):
                    self.helper = tf.contrib.seq2seq.TrainingHelper(self.dec_embed_input, self.target_lengths_)
                    self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                                   self.helper,
                                                                   self.enc_state,
                                                                   self.output_layer)

                    # unrolling the decoder layer
                    self.training_output, _, _ = tf.contrib.seq2seq.dynamic_decode(self.decoder,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=self.max_target_length_)

                # use this part of the graph for inferring unseen sentences
                with tf.variable_scope("inference"):
                    self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                                           tf.fill([self.params.batch_size],
                                                                                   language.word2idx["<start>"]),
                                                                           language.word2idx["<end>"])

                    if params.beam_search == 'n':
                        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                                       self.helper,
                                                                       self.enc_state,
                                                                       self.output_layer)
                    else:
                        # make beam width copies of the initial state to start the search
                        self.decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                            self.enc_state, multiplier=self.params.beam_width)

                        # initialize beam decoder
                        self.decoder = tf.contrib.seq2seq.BeamDecoder(self.decoder_cell,
                                                                      self.helper,
                                                                      tf.fill([self.params.batch_size],
                                                                              language.word2idx["<start>"]),
                                                                      self.language.word2idx["<end>"],
                                                                      self.decoder_initial_state,
                                                                      beam_width=params.beam_width,
                                                                      output_layer=self.output_layer,
                                                                      length_penalty_weight=0,
                                                                      coverage_penalty_weight=0)

                    # unrolling the decoder layer
                    self.inference_output, _, _  = tf.contrib.seq2seq.dynamic_decode(self.decoder,
                                                                                     impute_finished=True,
                                                                                     maximum_iterations=self.max_target_length_)

            with tf.name_scope("loss"):
                # Create masks that mimic the shapes of the original target sequences, effectively ignoring padding
                self.masks = tf.sequence_mask(self.target_lengths_, self.max_target_length_,
                                               dtype=tf.float32, name='masks')
                # Calculate cross_entropy loss of model output and expected output
                # train_output is a tuple of (logits, argmax(logits).index), dereferenced to obtained logits
                self.loss = tf.contrib.seq2seq.sequence_loss(self.training_output[0], self.targets_, self.masks)
            with tf.name_scope("optimize"):
                self.train_op = tf.train.RMSPropOptimizer(self.params.learning_rate).minimize(self.loss)

            with tf.name_scope("RL_loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inference_output[0], labels=self.actions_)
                self.RL_loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("RL_train"):
                # call this operation to train the model using the loss value
                self.RL_train_op = tf.train.RMSPropOptimizer(self.params.learning_rate).minimize(self.loss)


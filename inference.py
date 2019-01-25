import tensorflow as tf
import numpy as np

from keras import Input, Model
from keras.layers import Embedding, Dense, LSTM, CuDNNLSTM, GRU
from keras.models import load_model


def build_inference_models(units,embedding_dim,vocab_size, weights):
    encoder_weights = []
    encoder_weights.append(weights[0])
    encoder_weights += weights[1:4]
    decoder_weights = []
    decoder_weights.append(weights[0])
    decoder_weights += weights[4:9]
    # separate encoder and decoder models to use for testing/inference, initialize using weights from model
    # encoder model
    encoder_input = Input(shape=(None,), name="Encoder_input")
    encoder = LSTM(units, return_state=True, name='Encoder_lstm')
    encoder_embedding = Embedding(output_dim=embedding_dim, input_dim=vocab_size, name="Embedding")
    word_embedding_context = encoder_embedding(encoder_input)
    encoder_output, state_h, state_c = encoder(word_embedding_context)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_input, encoder_states)
    # decoder model
    decoder_input = Input(shape=(None,), name="Decoder_input")
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="Decoder_lstm")
    decoder_embedding = Embedding(output_dim=embedding_dim, input_dim=vocab_size, name="Embedding")
    word_embedding_answer = decoder_embedding(decoder_input)
    # keep states as inputs to loop through lstm cell
    decoder_state_input_h = Input(shape=(units,), name="H_state_input")
    decoder_state_input_c = Input(shape=(units,), name="C_state_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_output, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(vocab_size, activation='softmax', name="Dense_layer")
    decoder_output = decoder_dense(decoder_output)
    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_output] + decoder_states)
    # initialize weights using those from trained model
    encoder_model.set_weights(encoder_weights)
    decoder_model.set_weights(decoder_weights)
    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, language, max_length):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Populate the first character of target sequence with the start character.
    target_seq = np.array([language.word2idx["<start>"]])

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = language.idx2word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<end>' or
                    len(decoded_sentence) > max_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.array([sampled_token_index])

        # Update states
        states_value = [h, c]

    return decoded_sentence

from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time
import argparse
import random

import forwardprocessing
import processing
import models
import args

print(tf.__version__)

parser = argparse.ArgumentParser()
args.format_parser(parser)

parameters, unparsed = parser.parse_known_args()

path_to_file = parameters.train_file
print(path_to_file)


#######DATA PREPROCESSING#######

# Try experimenting with the size of that dataset
num_examples = parameters.num_examples
input_tensor, target_tensor_in, language, max_length = forwardprocessing.load_dataset(path_to_file, num_examples, parameters.vocab_file)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_in_train, target_tensor_in_val = train_test_split(input_tensor, target_tensor_in, test_size=parameters.test_size)

# Show length
print(len(input_tensor_train), len(target_tensor_in_train), len(input_tensor_val), len(target_tensor_in_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = parameters.batch_size
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = parameters.embedding_dim
units = parameters.units
vocab_size = len(language.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_in_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = models.Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = models.Decoder_attn(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
    # Mask out padding
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

checkpoint_dir = parameters.checkpoint_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

if parameters.continue_training == 'y':
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

EPOCHS = parameters.epochs

# accepts a tensor shape=(270,) of token indices and passes it through the model
# output in string token form
def evaluate(sentence, encoder, decoder, lang, max_length):
    inputs = tf.convert_to_tensor(sentence)
    inputs = tf.expand_dims(inputs, 0)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([lang.word2idx['<start>']], 0)

    for t in range(max_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()

        result += lang.idx2word[predicted_id] + ' '

        if lang.idx2word[predicted_id] == '<end>':
            return result, processing.idx_to_sentence(sentence, lang)

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, processing.idx_to_sentence(sentence, lang)

# accepts two strings representing consecutive turns of dialogue, and generates the models response to those turns
# output in string token form
def translate(sentence1, sentence2, encoder, decoder, lang, max_length):
    inputs = processing.sentence_to_idx(sentence1, lang, max_length)
    inputs = np.concatenate((inputs, processing.sentence_to_idx(sentence2, lang, max_length)), axis=None)
    return evaluate(inputs, encoder, decoder, lang, max_length)

def model_test(encoder, decoder, language, max_length):
    print("Performance on training set:")
    sentence = random.choice(input_tensor_train)
    result, sentence = evaluate(sentence, encoder, decoder, language, max_length)
    print("In: {}   Out: {}".format(sentence, result))
    print()
    print("Performance on test set:")
    sentence = random.choice(input_tensor_val)
    result, sentence = evaluate(sentence, encoder, decoder, language, max_length)
    print("In: {}   Out: {}".format(sentence, result))

for epoch in range(EPOCHS):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([language.word2idx['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        total_loss += batch_loss

        variables = encoder.variables + decoder.variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 75 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 1 == 0:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=encoder,
                                         decoder=decoder)
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("Saved at epoch {0}".format(epoch))
        model_test(encoder, decoder, language, max_length)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



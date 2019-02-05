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
num_examples = 324401
input_tensor, target_tensor_in, target_tensor_out, language, max_length = processing.load_dataset(path_to_file, num_examples, parameters.vocab_file)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_in_train, target_tensor_in_val, target_tensor_out_train, target_tensor_out_val = train_test_split(input_tensor, target_tensor_in, target_tensor_out, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_in_train), len(target_tensor_out_train), len(input_tensor_val), len(target_tensor_in_val), len(target_tensor_out_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 128
units = 256
vocab_size = len(language.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_in_train, target_tensor_out_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = models.Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = models.Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

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
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ_in, targ_out)) in enumerate(dataset):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)

            dec_hidden = enc_hidden

            #dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
            '''
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            '''
            predictions, dec_hidden, _ = decoder(targ_in, dec_hidden, enc_output)
            loss += loss_function(targ_out, predictions)

        batch_loss = (loss / int(targ_in.shape[1]))

        total_loss += batch_loss

        variables = encoder.variables + decoder.variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 75 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("Saved at Epoch {} Batch {}".format(epoch + 1, batch))

    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("Saved at epoch {0}".format(epoch))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence, encoder, decoder, lang, max_length):
    attention_plot = np.zeros((max_length, max_length))

    sentence = processing.preprocess_sentence(sentence)

    inputs = [lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([lang.word2idx['<start>']], 0)

    for t in range(max_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()

        result += lang.idx2word[predicted_id] + ' '

        if lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,
                                                max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

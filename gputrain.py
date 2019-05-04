from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
import pickle
import argparse

import processing
from processing import LanguageIndex
from models import seq2seq_model
import args

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

parser = argparse.ArgumentParser()
args.format_parser(parser)

parameters, unparsed = parser.parse_known_args()


def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')

    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)

    return inputs, targets, target_sequence_length, max_target_len

def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return lr_rate, keep_prob

def load_preprocess(datafile):
    with open(datafile, mode='rb') as in_file:
        return pickle.load(in_file)

save_path = parameters.checkpoint_dir

(encoder_input, decoder_input, decoder_output) = load_preprocess(parameters.train_file)
(encoder_lengths, decoder_lengths, decoder_lengths2) = load_preprocess(parameters.length_file)
vocab = processing.LanguageIndex
with open(parameters.vocab_file, mode='rb') as in_file:
    vocab = pickle.load(in_file)

max_target_sentence_length = max([len(sentence) for sentence in decoder_input])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs()
    lr, keep_prob = hyperparam_inputs()

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]), # todo: find out why the data is reversed, check the tutorial pages
                                                   targets,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   parameters,
                                                   vocab)

    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return tf.keras.preprocessing.sequence.pad_sequences(sentence_batch,
                                                                  maxlen=max_sentence)


def get_batches(sources, targets, batch_size, pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
# todo: there has to be a far more efficient built-in for this
#   only thing to determine is what to mark as "target", what makes the most sense would be decoder_output surely
train_source = encoder_input[parameters.batch_size:]
train_target = decoder_output[parameters.batch_size:]
valid_source = encoder_input[:parameters.batch_size]
valid_target = decoder_output[:parameters.batch_size]

(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             parameters.batch_size,
                                                                                                             vocab.word2idx["<pad>"]))
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(parameters.epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, parameters.batch_size,
                            vocab.word2idx["<pad>"])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: parameters.learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: parameters.keep_prob})
            # todo: marked incase something's wrong, lr and keep_prob here I'm assuming came from global variables


            if batch_i % parameters.display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    training_logits,
                    {input_data: source_batch,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)
                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(encoder_input) // parameters.batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')


def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

# Save parameters for checkpoint
save_params(save_path)



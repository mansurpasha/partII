from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
import pickle
import argparse

import processing
from processing import LanguageIndex, load_preprocess
import models
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

save_path = parameters.checkpoint_dir

(encoder_input, decoder_input, decoder_output) = load_preprocess(parameters.train_file)
#(encoder_lengths, decoder_lengths, decoder_lengths2) = load_preprocess(parameters.length_file)
vocab = processing.LanguageIndex
with open(parameters.vocab_file, mode='rb') as in_file:
    vocab = pickle.load(in_file)

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

# Reset the graph
tf.reset_default_graph()

# Instantiate the PGNetwork
Seq2SeqModel = models.Seq2Seq(params=parameters, language=vocab)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(max_to_keep=5)
if parameters.continue_training == 'y':
    saver.restore(sess, save_path)
    print("successfully loaded")

writer = tf.summary.FileWriter(parameters.tensorboard_dir)
writer.add_graph(sess.graph)

## Losses
tf.summary.scalar("Loss", Seq2SeqModel.loss)

## Reward mean
# tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_ )

write_op = tf.summary.merge_all()

for epoch_i in range(parameters.epochs):
    for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
            get_batches(train_source, train_target, parameters.batch_size,
                        vocab.word2idx["<pad>"])):

        _, loss = sess.run(
            [Seq2SeqModel.train_op, Seq2SeqModel.loss],
            {Seq2SeqModel.inputs_: source_batch,
             Seq2SeqModel.targets_: target_batch,
             Seq2SeqModel.target_lengths_: targets_lengths,
             Seq2SeqModel.max_target_length_: max(targets_lengths)})

        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={Seq2SeqModel.inputs_: source_batch,
                                                Seq2SeqModel.targets_: target_batch,
                                                Seq2SeqModel.target_lengths_: targets_lengths,
                                                Seq2SeqModel.max_target_length_: max(targets_lengths)})

        # summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
        writer.add_summary(summary, epoch_i * parameters.batch_size + batch_i)
        writer.flush()

    # Save Model
    saver.save(sess, save_path, epoch_i)
    # Evaluate Model
    batch_train_logits = sess.run(
        Seq2SeqModel.training_output[1],
        {Seq2SeqModel.inputs_: source_batch,
         Seq2SeqModel.targets_: target_batch,
         Seq2SeqModel.target_lengths_: targets_lengths,
         Seq2SeqModel.max_target_length_: max(targets_lengths)})

    batch_valid_logits = sess.run(
        Seq2SeqModel.inference_output[1],
        {Seq2SeqModel.inputs_: valid_sources_batch,
         Seq2SeqModel.max_target_length_: max(targets_lengths)})

    train_acc = get_accuracy(target_batch, batch_train_logits)
    valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

    print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
          .format(epoch_i, batch_i, len(encoder_input) // parameters.batch_size, train_acc, valid_acc, loss))

def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)

def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

# Save parameters for checkpoint

'''
# Optimizer
optimizer = tf.train.AdamOptimizer(lr)

# Gradient Clipping
gradients = optimizer.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(capped_gradients)
'''


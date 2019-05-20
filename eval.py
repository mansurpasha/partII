import tensorflow as tf
import numpy as np
import argparse

import models
import train
import args
import utils
import nltk

parser = argparse.ArgumentParser()
args.format_parser(parser)

parameters, unparsed = parser.parse_known_args()

# takes in a list of integer sequences (representing sentences) the output of a model
# and calculates the diversity value
def calculate_diversity(sentences):
    total_count = 0
    uniques = {}
    bi_uniques = {}
    for sentence in sentences:
        for word in sentence:
            total_count += 1
            if (word not in uniques.keys()):
                uniques[word] = 1
        for bigram in nltk.bigrams(sentence):
            if (bigram not in bi_uniques(sentence)):
                bi_uniques[bigram] = 1
    unigrams = len(uniques)/total_count
    bigrams = len(bi_uniques)/total_count
    return unigrams, bigrams

# takes in a list of integer sequences (representing sentences) the output of a model
# and calculates the average utterance length
def calculate_length_of_dialogues(sentences):
    return

# takes in a list of integer sequences (representing sentences) the output of a model
# and counts the 10 most common, store as dull sentences
def generate_dull_sentences(sentences):
    uniques = {}
    for sentence in sentences:
            if (sentence in uniques.keys()):
                uniques[sentence] = uniques[sentence] + 1
            else:
                uniques[sentence] = 1
    word_counts = sorted(uniques.items(), key=lambda kv: -kv[1])
    dull_responses = [x[0] for x in uniques[0:10]]
    return dull_responses

# given a file of input sentences, generate a response for each of them and store
def test_model(input_file, model, vocab):
    outputs = []
    sess = tf.Session()
    input = utils.load_preprocess(input_file)
    for batch_i, (source_batch, _, sources_lengths, targets_lengths) in enumerate(
            train.get_batches(input, input, parameters.batch_size, vocab.word2idx["<pad>"])):
        batch_outputs = sess.run(model.inference_output[1],
            {model.inputs_: source_batch,
             model.max_target_length_: parameters.max_target_length})
        for output in batch_outputs:
            outputs.append(output)
    return outputs

# given a model, and an input file name, run the model on the inputs,
# then calculate the diversity and length of dialogues metrics
def evaluate(input_file, model, vocab):
    outputs = (input_file, model, vocab)
    uni_diversity, bi_diversity = calculate_diversity(outputs)
    lengths = calculate_diversity(outputs)
    return uni_diversity, bi_diversity, lengths

# run evaluation process on any trained model by calling this script from the command line and feeding
# it the model's scope name (and vocab and input file)
if __name__ == "__main__":
    input_file = parameters.eval_file
    vocab = utils.load_preprocess(parameters.vocab_file)
    saver = tf.train.Saver
    model = models.RLModel(language=vocab, name="Eval", params=parameters)
    tf.train.init_from_checkpoint(parameters.checkpoint_dir, {parameters.old_scope: "Eval"})
    print(evaluate(input_file, model, vocab))


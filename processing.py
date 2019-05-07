from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
import unicodedata
import os
import numpy as np
import pickle

from nltk.tokenize import word_tokenize


def preprocess_sentence(w):
    return word_tokenize(w.lower())

def create_dataset(path, num_examples):
    # read datafile
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    # split datafile into lines, splits lines into sentences using separator token $$--$$,
    # and split sentences into lists of lowercase tokens
    sentence_sets = [[preprocess_sentence(w) for w in l.split('$$--$$')] for l in lines]

    return sentence_sets

# class that stores information on converting words to tokens and vice versa
class LanguageIndex():
    def __init__(self, vocab):
        self.word2idx = {}
        self.idx2word = {}

        self.set_index(vocab)

    def set_index(self, vocab):
        for (i, word) in enumerate(vocab):
            self.word2idx[word] = i
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def max_length(tensor):
    return max(len(t) for t in tensor)

# accepts a string of any format, and returns its equivalent as a padded numpy array
# output: numpy array of length max_len
def sentence_to_idx(sentence, lang, max_len):
    indices = [lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in word_tokenize(sentence.lower())]
    padded = tf.keras.preprocessing.sequence.pad_sequences([indices] ,
                                                           maxlen=max_len,
                                                           padding='post')
    return padded

# accepts a (padded) array of index tokens and returns the string equivalent, filtering out any padding
# output: string token format
def idx_to_sentence(sentence, lang):
    output = ""
    for word in sentence:
        if lang.idx2word[word] != "<pad>":
            output += ' ' + lang.idx2word[word]
    return output[1:]

# preprocess dataset into vectorized form
# arguments:
#   path - path to dataset
#   num_examples - number of lines to read from dataset file
#   path_to_vocab - path to precomputed vocab file
#   model_type - determines what source and target vectors will be, string either "prev2", "forward", or "backward"
#                prev2 - 2 turns of input, 1 turn of output
#                forward - 1 turn of input, 1 turn of output
#                backward - 1 turn of input, 1 turn of output, reversed source and target
# IMPORTANT: forward and backward to be used with different dataset file, namely input-target pairs, instead of triples
def load_dataset(path, num_examples, path_to_vocab, model_type):
    # creating cleaned input, output pairs
    triples = create_dataset(path, num_examples)

    # index language using the class defined above
    vocab = [w.strip("\n") for w in open(path_to_vocab, 'r').readlines()]

    lang = LanguageIndex(vocab)

    # vectorize input and target sentences, based on model being trained
    # base model that considers previous 2 turns of dialogue
    if model_type == "prev2":
        input_tensor = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp1] for inp1, inp2, targ in triples]
        input_tensor2 = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp2] for inp1, inp2, targ in triples]
        input_tensor = [np.concatenate((x,y)) for x, y in zip(input_tensor, input_tensor2)]
        decoder_input = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in targ] for inp1, inp2, targ in triples]
    # model that considers 1 turn of dialogue
    elif model_type == "forward":
        input_tensor = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp1] for inp1, targ in triples]
        decoder_input = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in targ] for inp1, targ in triples]
    # model that reverses source and target sentences
    elif model_type == "backward":
        decoder_input = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp1] for inp1, targ in triples]
        input_tensor = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in targ] for inp1, targ in triples]
    else:
        raise ValueError("correct model_type not found, use prev2, forward, or backward")

    # Add end of sentence markers to target_tensors. This signals when our decoder can stop generating a sequence
    # Seperate decoder_input and decoder_target (input drops the end marker and has a start marker)
    decoder_output = decoder_input.copy()
    for s in decoder_input:
        s.insert(0, lang.word2idx["<start>"])
    for s in decoder_output:
        s.append(lang.word2idx["<end>"])

    return input_tensor, decoder_input, decoder_output, lang

def create_vocab(path, num_examples, path_to_vocab):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above
    lang = [inp for inp, targ in pairs]
    lang.append(pairs[-1][1])

    word2idx = {}
    words = {}
    for sentence in lang:
        for word in sentence:
            if (word in words.keys()):
                words[word] = words[word] + 1
            else:
                words[word] = 1
    word_counts = sorted(words.items(), key=lambda kv: -kv[1])
    top_words = [x[0] for x in word_counts[0:10000]]

    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    word2idx['<unk>'] = 3
    word2idx['start_of_conversation_token'] = 4

    for index, word in enumerate(top_words):
        word2idx[word] = index + 4

    f = open(path_to_vocab, 'w')
    for k in word2idx:
        f.write(k + "\n")
    f.close()

def load_preprocess(datafile):
    with open(datafile, mode='rb') as in_file:
        return pickle.load(in_file)

def generate_seq_lengths(sequences):
    return [len(sequence) for sequence in sequences]

if __name__ == "__main__":
    # generate vocab file
    '''create_vocab("/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/data/self_dialogue_corpus/processed/nples.txt",
             324401, "vocab_file")'''
    # pickle and store all preprocessed (vectorized) data and pickled vocab class
    '''
    prev2_filepath = "/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/data/self_dialogue_corpus/processed/2ples.txt"
    prev1_filepath = "/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/data/self_dialogue_corpus/processed/nples.txt"
    vocab_file = "/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/vocab_file"

    p2_encoder_input, p2_decoder_input, p2_decoder_output, _ = load_dataset(prev2_filepath, 324401, vocab_file, "prev2")
    pickle.dump((p2_encoder_input, p2_decoder_input, p2_decoder_output), open('preprocess_prev2.p', 'wb'))
    print("written p2")

    f_encoder_input, f_decoder_input, f_decoder_output, _ = load_dataset(prev1_filepath, 324401, vocab_file, "forward")
    pickle.dump((f_encoder_input, f_decoder_input, f_decoder_output), open('preprocess_forward.p', 'wb'))
    print("written forward")

    b_encoder_input, b_decoder_input, b_decoder_output, vocab = load_dataset(prev1_filepath, 324401, vocab_file, "backward")
    pickle.dump((b_encoder_input, b_decoder_input, b_decoder_output), open('preprocess_backward.p', 'wb'))
    print("written backward")

    pickle.dump(vocab, open('vocab.p', 'wb'))
    print("written vocab")
    
    # append sequence lengths to our pickles
    with open("pickles/preprocess_prev2.p", mode='rb') as in_file:
        p2_enc_inp, p2_dec_inp, p2_dec_out = pickle.load(in_file)
    p2_enc_inp_lengths = generate_seq_lengths(p2_enc_inp)
    p2_dec_inp_lengths = generate_seq_lengths(p2_dec_inp)
    p2_dec_out_lengths = generate_seq_lengths(p2_dec_out)
    pickle.dump((p2_enc_inp_lengths, p2_dec_inp_lengths, p2_dec_out_lengths), open('lengths_prev2.p', 'wb'))
    with open("pickles/preprocess_forward.p", mode='rb') as in_file:
        f_enc_inp, f_dec_inp, f_dec_out = pickle.load(in_file)
    f_enc_inp_lengths = generate_seq_lengths(f_enc_inp)
    f_dec_inp_lengths = generate_seq_lengths(f_dec_inp)
    f_dec_out_lengths = generate_seq_lengths(f_dec_out)
    pickle.dump((f_enc_inp_lengths, f_dec_inp_lengths, f_dec_out_lengths), open('lengths_forward.p', 'wb'))
    with open("pickles/preprocess_backward.p", mode='rb') as in_file:
        b_enc_inp, b_dec_inp, b_dec_out = pickle.load(in_file)
    b_enc_inp_lengths = generate_seq_lengths(b_enc_inp)
    b_dec_inp_lengths = generate_seq_lengths(b_dec_inp)
    b_dec_out_lengths = generate_seq_lengths(b_dec_out)
    pickle.dump((b_enc_inp_lengths, b_dec_inp_lengths, b_dec_out_lengths), open('lengths_backward.p', 'wb'))
    print("written all lenghts")
    '''
    with open("pickles/lengths_prev2.p", mode='rb') as in_file:
        enc, dec_in, dec_out = pickle.load(in_file)
    print(dec_in == dec_out)
    print(enc[0:30])
    print(dec_in[0:30])
    print(dec_out[0:30])

    bins = np.bincount(dec_in)
    for i, x in enumerate(bins):
        print("{}: {}".format(i, x))


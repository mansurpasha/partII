from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
import unicodedata
import os
import numpy as np

from nltk.tokenize import word_tokenize


def preprocess_sentence(w):
    return word_tokenize(w.lower())

def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('$$--$$')] for l in lines[:num_examples]]

    return word_pairs

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

def sentence_to_idx(sentence, lang, max_len):
    indices = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in word_tokenize(sentence)]]
    padded = tf.keras.preprocessing.sequence.pad_sequences(indices,
                                                           maxlen=max_len,
                                                           padding='post')
    return padded

def load_dataset(path, num_examples, path_to_vocab):
    # creating cleaned input, output pairs
    triples = create_dataset(path, num_examples)

    # index language using the class defined above
    vocab = [w.strip("\n") for w in open(path_to_vocab, 'r').readlines()]

    lang = LanguageIndex(vocab)

    # Vectorize the input and target languages
    input_tensor = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp1] for inp1, inp2, targ in triples]
    input_tensor2 = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp2] for inp1, inp2, targ in triples]
    decoder_input = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in targ] for inp1, inp2, targ in triples]

    # Add start and end of sentence markers to respective sentences, creating two sets for decoder input and output
    for s in input_tensor:
        s.append(lang.word2idx["<end>"])
    for s in input_tensor2:
        s.append(lang.word2idx["<end>"])
    for s in decoder_input:
        s.insert(0,lang.word2idx["<start>"])

    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_seq_length = max(max_length(input_tensor), max_length(decoder_input))

    # Padding the input and output tensor to the maximum length
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_seq_length,
                                                                 padding='post')

    encoder_input2 = tf.keras.preprocessing.sequence.pad_sequences(input_tensor2,
                                                                 maxlen=max_seq_length,
                                                                 padding='post')

    decoder_input = tf.keras.preprocessing.sequence.pad_sequences(decoder_input,
                                                                  maxlen=max_seq_length,
                                                                  padding='post')

    encoder_input = [np.concatenate((x,y)) for x, y in zip(encoder_input, encoder_input2)]

    return encoder_input, decoder_input, lang, max_seq_length

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

    for index, word in enumerate(top_words):
        word2idx[word] = index + 4

    f = open(path_to_vocab, 'w')
    for k in word2idx:
        f.write(k + "\n")
    f.close()

if __name__ == "__main__":
    create_vocab("/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/data/self_dialogue_corpus/processed/nples.txt",
             324401, "vocab_file")

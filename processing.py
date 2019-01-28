from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
import unicodedata
import os
import numpy as np

from nltk.tokenize import word_tokenize




# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):

    '''
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()
    '''
    return word_tokenize(w.lower())


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('$$--$$')] for l in lines[:num_examples]]

    return word_pairs


# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        words = {}
        for sentence in self.lang:
            for word in sentence:
                if (word in words.keys()):
                    words[word] = words[word] + 1
                else:
                    words[word] = 1
        word_counts = sorted(words.items(), key=lambda kv: -kv[1])
        top_words = [x[0] for x in word_counts[0:10000]]

        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<end>'] = 2
        self.word2idx['<unk>'] = 3

        for index, word in enumerate(top_words):
            self.word2idx[word] = index + 4

        for word, index in self.word2idx.items():
            self.idx2word[index] = word




def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above
    lang = [inp for inp, targ in pairs]
    lang.append(pairs[-1][1])
    lang = LanguageIndex(lang)

    # Vectorize the input and target languages
    input_tensor = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in inp] for inp, targ in pairs]
    decoder_input = [[lang.word2idx[s] if (s in lang.word2idx) else lang.word2idx['<unk>'] for s in targ] for inp, targ in pairs]
    decoder_output = decoder_input

    # Add start and end of sentence markers to respective sentences, creating two sets for decoder input and output
    for s in input_tensor:
        s.append(lang.word2idx["<end>"])
    for s in decoder_input:
        s.insert(0,lang.word2idx["<start>"])
    for s in decoder_output:
        s.append(lang.word2idx["<end>"])

    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_seq_length = max(max_length(input_tensor), max_length(decoder_input))

    # Padding the input and output tensor to the maximum length
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_seq_length,
                                                                 padding='post')

    decoder_input = tf.keras.preprocessing.sequence.pad_sequences(decoder_input,
                                                                  maxlen=max_seq_length,
                                                                  padding='post')

    decoder_output = tf.keras.preprocessing.sequence.pad_sequences(decoder_output,
                                                                  maxlen=max_seq_length,
                                                                  padding='post')

    one_hot = [[np.zeros(len(lang.word2idx)) for x in y] for y in decoder_output]
    one_hot = np.zeros((num_examples,max_seq_length,len(lang.word2idx)))
    for i in range(len(decoder_output)):
        for j in range(max_seq_length):
            index = decoder_output[i][j]
            one_hot[i][j][index] = 1

    return encoder_input, decoder_input, one_hot, lang, max_seq_length

'''
# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
'''
def create_vocab(path, num_examples, path_to_vocab):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above
    lang = [inp for inp, targ in pairs]
    lang.append(pairs[-1][1])
    lang = LanguageIndex(lang)

    f = open(path_to_vocab, 'w')
    for k in lang.word2idx:
        f.write(k)
    f.close()

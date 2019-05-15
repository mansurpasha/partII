"""
A dialogue system meant to be used for language learning.

This is based on Google Neural Machine Tranlation model
https://github.com/tensorflow/nmt
which is based on Thang Luong's thesis on
Neural Machine Translation: https://github.com/lmthang/thesis

And on the paper Building End-To-End Dialogue Systems
Using Generative Hierarchical Neural Network Models:
https://arxiv.org/pdf/1507.04808.pdf

Created by Tudor Paraschivescu for the Cambridge UROP project
"Dialogue systems for language learning"
Base methods for preprocessing the Cornell Movie-Dialogs dataset into conversations.

Code adapted from
https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/assignments/chatbot/data.py.
"""

import os
import sys
from os.path import join, curdir
sys.path.append('utils')
#from utils.preprocessing_utils import make_dir
#from utils.misc_utils import get_parent_dir

LINE_FILE = "/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/data/cornell_movie_dialogs_corpus/movie_lines.txt"
CONVO_FILE = "/Users/mansurpasha/map79/partII/Individual Project/DialogueSystem/data/cornell_movie_dialogs_corpus/movie_conversations.txt"

# string separators used to write processed tuples to file. These separators
# are picked up when using split() to read the files
SEP = "$$--$$"


def get_lines():
    id2line = {}
    file_path = LINE_FILE
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line


def get_convos():
    """ Get conversations from the raw data """
    file_path = CONVO_FILE
    convos = []
    with open(file_path, 'r', encoding='cp1252') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos


def build_conv(id2line, convos):
    """Create a matrix of conversations, each row representing a vector of lines"""
    conversations = []
    for convo in convos:
        current_conv = []
        for index, line in enumerate(convo):
            current_conv.append(id2line[convo[index]])
        conversations.append(current_conv)
    return conversations

def load_conversations():
    return build_conv(get_lines(), get_convos())

def conversation_to_nples(n, turns):
    tuples = []
    for i in range(len(turns) - n):
        source = turns[i:i + n - 1]
        target = turns[i + n - 1]
        tuples.append(turns[i:i + n])
    return tuples

def process_conversations(n, destination="processed_cornell"):
    conversations = load_conversations()
    nples = []
    for conversation in conversations:
        nples.append(conversation_to_nples(n, conversation))
    f = open(destination, "w")
    print(nples)
    for tuples in nples:
        for tuple in tuples:
            out = ""
            for i in range(len(tuple)-1):
                out += tuple[i] + SEP
            out += tuple[-1] + '\n'
            f.write(out)
    f.close

if __name__ == "__main__":
    process_conversations(3, join(curdir,"data/cornell_movie_dialogs_corpus/processed/prev2.txt"))
    process_conversations(2, join(curdir,"data/cornell_movie_dialogs_corpus/processed/forward.txt"))

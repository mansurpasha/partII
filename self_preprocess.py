import os
import unicodedata
import re
from os import listdir
from os.path import isfile, join, curdir
from tqdm import tqdm

# string separators used to write processed tuples to file. These separators
# are picked up when using split() to read the files
SEP = "$$--$$"
# token to represent an empty turn of conversation at the start of a dialogue
SOC_token = "start_of_conversation_token"

current_fp = os.path.realpath(__file__)

# takes a file of conversation turns and returns a target sentence
# defined by the n-1 previous turns
def conversation_to_nples(filename, n):
    f = open(filename, "r")
    turns = []
    for i in range(n-1):
        turns.append(SOC_token)
    for line in f:
        turns.append(line.strip('\n'))

    #create a source/target pairs using (1 to n-1th)/(nth) turns
    tuples = []
    for i in range(len(turns)-n):
        source = turns[i:i+n-1]
        target = turns[i+n-1]
        tuples.append(turns[i:i+n])
    return tuples

# takes the path to the directory containing the self dialogues
# and returns a file list of all sentence n-grams
def process_conversations(path, n, destination="processed_self"):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    nples = []
    for file in tqdm(filenames):
        for nplets in conversation_to_nples(join(path,file), n):
            nples.append(nplets)
    if not os.path.exists(destination):
        os.makedirs(destination)
    f = open(join(destination, "2ples.txt"), "w")
    for tuple in nples:
        out = ""
        for i in range(len(tuple)-1):
            out += tuple[i] + SEP
        out += tuple[len(tuple)-1] + '\n'
        f.write(out)
    f.close

def get_start_sentences(path, destination):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    init_sentences = [open(join(path,filename)).readline() for filename in filenames]
    f = open(join(destination, "init_sentences"), "w")
    for sentence in init_sentences:
        f.write(sentence)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

if __name__ == '__main__':
    process_conversations(join(curdir,"data/self_dialogue_corpus/dialogues"),
                      3, join(curdir,"data/self_dialogue_corpus/processed"))
    #get_start_sentences(join(curdir,"data/self_dialogue_corpus/dialogues"),
                        #join(curdir,"data/self_dialogue_corpus/processed"))


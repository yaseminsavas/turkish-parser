'''

Project Title: Graph - Based BiLSTM & Stack LSTM in Turkish
Author: Yasemin Savaş 54085

'''

# Libraries
from collections import Counter, OrderedDict
from src.utils import *

print("Gathering the data...")

train_directory = 'data/UD_Turkish-BOUN-master/tr_boun-ud-train.conllu'
validation_directory = 'data/UD_Turkish-BOUN-master/tr_boun-ud-dev.conllu'

# Preparing the data: words, ids, positions, and relations
words = Counter()
pos = list()
rel = list()

with open(train_directory, 'r') as file:
    for sentence in read_conll(file):
        words.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
        pos += [node.pos for node in sentence if isinstance(node, ConllEntry)]
        rel += [node.relation for node in sentence if isinstance(node, ConllEntry)]

word_ids = {w: i for i, w in enumerate(words.keys())}
position = list(OrderedDict.fromkeys(pos))
relations = list(OrderedDict.fromkeys(rel))

print("Data gathering is done.")
print(" ")

print("Defining the model...")
# TODO: Model definition
print("Model definition is done.")
print(" ")


print("Training the model...")
# TODO: Model training
print("Model training is done.")
print(" ")


# TODO:  Performance
print("Evaluating the performance...")
print(" ")

print("End of the project.")
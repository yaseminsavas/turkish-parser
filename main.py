'''

Project Title: Graph - Based BiLSTM & Stack LSTM Models in Turkish
Author: Yasemin Savaş - 54085

Codes related to the BiLSTM implementation are taken & adapted from here:

Paper Link: https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198
Github Repository: https://github.com/elikip/bist-parser

Note:
I used GloVe embeddings from here: https://github.com/inzva/Turkish-GloVe

'''

from collections import Counter, OrderedDict
from src.utils import *
import os
from os import path
from src import bilstm_model, stack_model
from src.main_utils import main_train

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

# TODO: Don't forget to use this
#if path.isdir(f"turkish-parser/results/") is False:
#    os.mkdir(f"turkish-parser/results")


runmode = "BiLSTM"  # Available arguments are BiLSTM or StackLSTM
print("Selected model type:", runmode)

if runmode == "BiLSTM":
    print("Defining the model...")
    model = bilstm_model.BiLSTM(words, position, relations, word_ids)
    print("Model definition is done.")
    print(" ")
    print("Training the model...")

    main_train(model, train_directory, validation_directory)

elif runmode == "StackLSTM":
    print("Defining the model...")
    print("Model definition is done.")
    model = stack_model.StackLSTM(words, position, relations, word_ids)
    print(" ")
    print("Training the model...")

    main_train(model, train_directory, validation_directory)

else:
    raise("Provide a valid model name!")

print("End of the project.")
print("See files under the results folder to see the epoch performances on the validation set.")

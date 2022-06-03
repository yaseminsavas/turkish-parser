'''

Project Title: Graph - Based BiLSTM in Turkish
Author: Yasemin Savaş - 54085

This project is based on this paper:
Paper Link: https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198
Github Repository: https://github.com/elikip/bist-parser

Notes:
1. I used GloVe embeddings from here: https://github.com/inzva/Turkish-GloVe

2. The implementation is based on PyTorch.

3. There are subtle differences between the original implementation and this repo.
(Learning rate is 0.001, MLP activations are ReLu, a weight decay exists, the loss is negative hinge embedding loss..)

'''

from collections import Counter, OrderedDict
from src.utils import *
import os
from src import bilstm_model

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
pos_tag = list(OrderedDict.fromkeys(pos))
rel_tag = list(OrderedDict.fromkeys(rel))

print("Data gathering is done.")
print(" ")

print("Defining the model...")
print("Model definition is done.")
model = bilstm_model.BiLSTM(words, pos_tag, rel_tag, word_ids)
print(" ")

print("Training the model...")
model.train(train_directory)
print(" ")

print("Predicting...")
predictions = model.predict(validation_directory)
write_conll(f"/Users/yaseminsavas/turkish-parser/results/dev_epoch_30.conllu",
            predictions)
print("Predictions are printed.")
print(" ")
# Evaluation with the validation data
path = f"/Users/yaseminsavas/turkish-parser/results/dev_epoch_30.conllu"
os.system(
    'python /Users/yaseminsavas/turkish-parser/src/evaluation_script/conll17_ud_eval.py -v -w /Users/yaseminsavas/turkish-parser/src/evaluation_script/weights.clas '
    + validation_directory + ' ' + path + ' > ' + path + '.txt')

print("Performance is evaluated.")
print(" ")

print("End of the project.")
print(" ")
print("See files under the results folder to see the epoch performances on the validation set.")

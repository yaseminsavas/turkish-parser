'''

Project Title: Graph - Based BiLSTM in Turkish using PyTorch
Author: Yasemin Savaş - 54085

This project is based on this paper:
Paper Link: https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198
Github Repository: https://github.com/elikip/bist-parser

Notes:
1. I used GloVe embeddings from here: https://github.com/inzva/Turkish-GloVe

2. The implementation is based on PyTorch.

3. There are some differences between the original implementation and this repo.
(Learning rate - MLP activations - weight decay - the loss criterion function...)

'''

from collections import Counter, OrderedDict
from src.utils import *
import os
from src import bilstm_model, trainer

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

print("Defining the model & trainer...")
model = bilstm_model.BiLSTM(words, pos_tag, rel_tag, word_ids)
trainer = trainer.Trainer(model)
print("Model & trainer definitions are done.")
print(" ")


print("Training the model...")
for i in range(30):
    print("EPOCH:", i+1)
    trainer.train(model, train_directory)
    print(" ")

    print("Predicting...")
    predictions = model.predict(validation_directory)
    write_conll(f"results/dev_epoch_{i+1}.conllu",
                predictions)
    print("Predictions are printed.")
    print(" ")
    # Evaluation with the validation data
    path = f"results/dev_epoch_{i+1}.conllu"
    os.system(
        'python src/evaluation_script/conll17_ud_eval.py -v -w src/evaluation_script/weights.clas '
        + validation_directory + ' ' + path + ' > ' + path + '.txt')

    print("Performance is evaluated.")
    print(" ")

print("End of the project.")
print(" ")
print("See files under the results folder to see the prediction performance on the validation set.")

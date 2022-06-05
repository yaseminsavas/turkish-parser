from src.utils import read_conll
import random
import src
import torch
from torch import optim, nn
import numpy as np


class Trainer:
    def __init__(self, model):

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        self.conll_path = 'data/UD_Turkish-BOUN-master/tr_boun-ud-train.conllu'

    def train(self, model, conll_path):
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)

            for index, sentence in enumerate(shuffledData):
                if index % 100 == 0 and index > 0:
                    print('Sentence:', index)

                scores, conll_sentence = model.forward(sentence)
                scores = np.max(scores, axis=2)

                gold = [entry.parent_id for entry in conll_sentence]
                heads = src.decoder.parse_proj(scores, gold)

                loss = self.criterion(torch.tensor(scores[1:]), torch.tensor(gold[1:]))
                loss.requires_grad = True
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

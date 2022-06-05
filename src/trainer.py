from src.utils import read_conll
import random
import src
import torch
from torch import autograd
from torch import optim, nn


class Trainer:
    def __init__(self, model):

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0005)
        self.criterion = nn.HingeEmbeddingLoss()
        self.conll_path = 'data/UD_Turkish-BOUN-master/tr_boun-ud-train.conllu'


    def train(self, model, conll_path):
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)

            for index, sentence in enumerate(shuffledData):
                if index % 100 == 0 and index > 0:
                    print('Sentence:', index, "Loss: ", loss.item())

                scores, conll_sentence = model.forward(sentence)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = src.decoder.parse_proj(scores, gold)

                self.optimizer.zero_grad()
                loss = self.criterion(torch.tensor(heads).float(), torch.tensor(gold).float())
                loss = autograd.Variable(loss, requires_grad=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                loss.backward()
                self.optimizer.step()
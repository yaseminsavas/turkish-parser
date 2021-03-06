from src.utils import read_conll
from src.main_utils import get_embedding
from operator import itemgetter
import src.utils, src.decoder
import numpy as np
from torch import autograd, nn
import torch


class BiLSTM(nn.Module):
    def __init__(self, words, pos_tag, rel_tag, word_ids):

        super(BiLSTM, self).__init__()

        self.bilstm = nn.LSTM(input_size=425,
                              hidden_size=100,
                              num_layers=1,
                              bidirectional=True)

        self.words = words
        self.word_ids = word_ids
        self.pos_tag = {pos: ind for ind, pos in enumerate(pos_tag)}
        self.rel_tag = {rel: ind for ind, rel in enumerate(rel_tag)}

        self.external_embedding = get_embedding("data/Glove_embeddings/vectors.txt")
        self.emb_dict = {word: embedding for word, embedding in enumerate(self.external_embedding)}
        self.extrnd = {word: i for i, word in enumerate(self.external_embedding)}

        self.word_emb = nn.Embedding(len(word_ids), 100)
        self.pos_emb = nn.Embedding(len(pos_tag), 25)
        self.ext_emb = nn.Embedding(len(self.external_embedding), 300)

        self.mlp_in = nn.Linear(in_features=625,
                                out_features=400)

        self.mlp_out = nn.Linear(in_features=400,
                                 out_features=42)

    def fw2(self, sentence, i, j):
        if sentence[i].headfov is None:
            sentence[i].headfov = self.mlp_in(torch.cat([sentence[i].lstms[0], sentence[i].lstms[1]], 2))
        if sentence[j].modfov is None:
            sentence[j].modfov = self.mlp_in(torch.cat([sentence[j].lstms[0], sentence[j].lstms[1]], 2))

        output = torch.tanh(self.mlp_out(sentence[i].headfov + sentence[j].modfov))

        return output

    def forward(self, sentence):

        conll_sentence = [entry for entry in sentence if isinstance(entry, src.utils.ConllEntry)]

        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):

            word_idxs = autograd.Variable(torch.LongTensor([self.words.get(entry.norm, 0)]))
            pos_idxs = autograd.Variable(torch.LongTensor([self.pos_tag[entry.pos]]))
            emb_idxs = autograd.Variable(torch.LongTensor([self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0))]))

            word_embeddings = self.word_emb(word_idxs)
            pos_embeddings = self.pos_emb(pos_idxs)
            ext_embeddings = self.ext_emb(emb_idxs)

            word_pos_cat = torch.cat((word_embeddings, pos_embeddings, ext_embeddings), 1).unsqueeze(1)

            entry.vec = word_pos_cat
            lstm_output, _ = self.bilstm(word_pos_cat)

            rword_idxs = autograd.Variable(torch.LongTensor([self.words.get(rentry.norm, 0)]))
            rpos_idxs = autograd.Variable(torch.LongTensor([self.pos_tag[rentry.pos]]))
            remb_idxs = autograd.Variable(
                torch.LongTensor([self.extrnd.get(rentry.form, self.extrnd.get(rentry.norm, 0))]))

            rword_embeddings = self.word_emb(rword_idxs)
            rpos_embeddings = self.pos_emb(rpos_idxs)
            rext_embeddings = self.ext_emb(remb_idxs)

            rword_pos_cat = torch.cat((rword_embeddings, rpos_embeddings, rext_embeddings), 1).unsqueeze(1)
            rentry.vec = rword_pos_cat

            rlstm_output, _ = self.bilstm(rword_pos_cat)

            entry.lstms = [word_pos_cat, lstm_output]
            rentry.lstms = [rlstm_output, rword_pos_cat]

            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

        exprs = [[self.fw2(conll_sentence, i, j) for j in range(0, len(conll_sentence))] for i in
                 range(0, len(conll_sentence))]

        scores = np.array([[output.detach().numpy()[0][0] for output in exprsRow] for exprsRow in exprs])

        return scores, conll_sentence

    def evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.mlp_in(torch.cat([sentence[i].lstms[0], sentence[i].lstms[1]], 2))
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = self.mlp_in(torch.cat([sentence[j].lstms[0], sentence[j].lstms[1]], 2))

        output = torch.tanh(self.mlp_out(sentence[i].rheadfov + sentence[j].rmodfov))

        return output.detach().numpy()[0][0], output

    def predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):

                scores, conll_sentence = self.forward(sentence)
                scores = np.max(scores,axis=2)
                heads = src.decoder.parse_proj(scores)

                for entry, head in zip(conll_sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                for modifier, head in enumerate(heads[1:]):
                    scores, exprs = self.evaluateLabel(conll_sentence, head, modifier + 1)

                    conll_sentence[modifier + 1].pred_relation = list(self.rel_tag)[max(enumerate(scores),
                                                                                        key=itemgetter(1))[0]]
                yield sentence

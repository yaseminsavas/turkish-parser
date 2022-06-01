from dynet import *
from src.utils import read_conll
from operator import itemgetter
import time, random, src.decoder
import numpy as np


class StackLSTM:
    def __init__(self, vocab, pos, rels, w2i):

        self.model = Model()
        self.trainer = AdamTrainer(self.model,alpha=0.05)
        self.trainer.learning_rate = 0.05

        self.words = vocab
        self.vocab = {word: ind for word, ind in w2i.items()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.external_embedding = "data/GloVe_embeddings/vectors.txt"

        external_embedding_fp = open("data/GloVe_embeddings/vectors.txt", 'r')
        external_embedding_fp.readline()
        self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                   external_embedding_fp}
        external_embedding_fp.close()

        self.edim = len(list(self.external_embedding.values())[0]) # 300
        self.noextrn = [0.0 for _ in range(0, 300)]
        self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
        self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, 300))
        for word, i in self.extrnd.items():
            self.elookup.init_row(i, self.external_embedding[word])

        self.builder = LSTMBuilder(1, 125+300, 125+300, self.model)

        self.wlookup = self.model.add_lookup_parameters((len(vocab), 100))
        self.plookup = self.model.add_lookup_parameters((len(pos), 25))
        self.rlookup = self.model.add_lookup_parameters((len(rels), 25))

        self.hidLayerFOH = self.model.add_parameters((100, (125+300) * 2))
        self.hidLayerFOM = self.model.add_parameters((100, (125+300) * 2))
        self.hidBias = self.model.add_parameters(100)

        self.outLayer = self.model.add_parameters((1, 100))

        self.rhidLayerFOH = self.model.add_parameters((100, 2 * (125+300)))
        self.rhidLayerFOM = self.model.add_parameters((100, 2 * (125+300)))
        self.rhidBias = self.model.add_parameters((100))

        self.routLayer = self.model.add_parameters((len(self.irels), 100))
        self.routBias = self.model.add_parameters((len(self.irels)))

    def  __getExpr(self, sentence, i, j):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov = self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        output = self.outLayer.expr() * rectify(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr())

        return output

    def __evaluate(self, sentence, train):
        exprs = [ [self.__getExpr(sentence, i, j) for j in range(len(sentence))] for i in range(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])

        return scores, exprs

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        output = self.routLayer.expr() * rectify(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

        return output.value(), output

    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, src.utils.ConllEntry)]

                for entry in conll_sentence:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))]
                    posvec = self.plookup[int(self.pos[entry.pos])]
                    evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm,0)))]
                    entry.vec = concatenate([wordvec, posvec, evec])

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                lstm_stack = self.builder.initial_state()

                for entry in conll_sentence:
                    lstm_stack2 = lstm_stack.add_input(entry.vec)
                    lstm_stack3 = lstm_stack2.add_input(entry.vec)
                    lstm_stack4 = lstm_stack3.add_input(entry.vec)

                    entry.lstms[1] = lstm_stack4.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                heads = src.decoder.parse_proj(scores)

                for entry, head in zip(conll_sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                for modifier, head in enumerate(heads[1:]):
                    scores, exprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                    conll_sentence[modifier+1].pred_relation = list(self.irels)[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                yield sentence

    def Train(self, conll_path,counter):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            data = list(read_conll(conllFP))

            errs = []
            lerrs = []
            eeloss = 0.0

            for index, sentence in enumerate(data):
                if index % 100 == 0 and index != 0:
                    print('Processing sentence number:', index, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal)

                conll_sentence = [entry for entry in sentence if isinstance(entry, src.utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.words.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0]
                    posvec = self.plookup[int(self.pos[entry.pos])] if 25 > 0 else None

                    evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (
                                dropFlag or (random.random() < 0.5)) else 0]
                    entry.vec = concatenate([wordvec,posvec,evec])

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                lstm_stack = self.builder.initial_state()

                for entry in conll_sentence:
                    lstm_stack2 = lstm_stack.add_input(entry.vec)
                    lstm_stack3 = lstm_stack2.add_input(entry.vec)
                    lstm_stack4 = lstm_stack3.add_input(entry.vec)

                    entry.lstms[1] = lstm_stack4.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = src.decoder.parse_proj(scores, gold)

                for modifier, head in enumerate(gold[1:]):
                    rscores, rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                    goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                    wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                    if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                        lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if index % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []

                    renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()

        self.trainer.update()
        #weight-decay
        self.trainer.learning_rate = self.trainer.learning_rate / (1-0.95)
        print("Loss: ", mloss/index)
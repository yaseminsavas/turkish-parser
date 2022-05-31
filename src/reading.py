from conllu import TokenList
from collections import OrderedDict

def compile_sent(sent):
    sent_list = TokenList()
    for i, tok_data in enumerate(sent):
        tok_id = i + 1
        tok = tok_data[0]
        lemma = tok_data[1]
        pos = tok_data[2]
        feats = tok_data[3]
        compiled_tok = OrderedDict({'id': tok_id, 'form': tok, 'lemma': lemma, 'upostag': pos, 'xpostag': None, 'feats': feats, 'head': None, 'deprel': None, 'deps': None, 'misc': None})
        sent_list.append(compiled_tok)
    sent_list = sent_list.serialize()
    return sent_list
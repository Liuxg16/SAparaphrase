# -*- coding: utf-8 -*-
import numpy as np
import sys, string
import os,re
from collections import deque
from nltk.translate.bleu_score import corpus_bleu


bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}


def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()
    for i in range(len(bleu_score_weights)):
        bleu_scores[i + 1] = round(
            corpus_bleu(
                list_of_references=actual_word_lists[:len(generated_word_lists)],
                hypotheses=generated_word_lists,
                weights=bleu_score_weights[i + 1]), 4)

    return bleu_scores

def clarify(line):
    # line = re.sub(r'\d+',' sss ', line)
    printable = set(string.printable)
    line = filter(lambda x: x in printable, line)
    line = line.replace('?',' ')
    line = line.replace('(',' ')
    line = line.replace(']',' ')
    line = line.replace('[',' ')
    line = line.replace('{',' ')
    line = line.replace('}',' ')
    line = line.replace(')',' ')
    line = line.replace('-',' ')
    line = line.replace('!',' ')
    line = line.replace('.',' ')
    line = line.replace(',',' ')
    line = line.replace(';',' ')
    line = line.replace('\'',' \' ')
    line = line.replace('\' s',' \'s')
    line = line.replace('\' t',' \'t')
    line = line.replace('"',' ')

    return line

def savetexts(sent_list, file_name):
    # list(list(word))
    fileobject = open(file_name, 'w')
    for sent in sent_list:
        fileobject.write(' '.join(sent))
        fileobject.write('\n')
    fileobject.close()

def appendtext(text, file_name):
    # list(list(word))
    fileobject = open(file_name, 'a+')
    fileobject.write(' '.join(text))
    fileobject.write('\n')
    fileobject.close()


if __name__ == "__main__":
    sent = 'I 地方have999 a33)) pretty-computer.'
    print(clarify(sent))

# -*- coding: utf-8 -*-
'''
Created on Apr 11, 2016

@author: hugohrosa
'''
import codecs
import pickle

sent = {'N0_Neg':-1,'N1_Neg':-1,'N0_Pos':1,'N1_Pos':1,'N0_Neu':0,'N1_Neu':0}
out = dict()
for l in codecs.open('DATA/Palavras Simples com Polaridade.txt','r','utf-16'):
    sent_score = 0
    sent_lbl = ''
    ls = l.split('.')
    words = ls[0].split(',')
    for s in sent.keys():
        if s in ls[1]:
            sent_score += sent[s]
    if sent_score < 0:
        sent_lbl = 'NEG'
    elif sent_score == 0:
        sent_lbl = 'NEU'
    elif sent_score > 0:
        sent_lbl = 'POS'
    for w in words:
        out[w]=sent_lbl

with open('DATA/sentilex.pkl',"wb") as fo:
    pickle.dump(out,fo)



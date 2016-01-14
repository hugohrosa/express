# -*- coding: utf-8 -*-
import codecs
from ipdb import set_trace
import numpy as np
import cPickle
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import math
import itertools
import pprint
import re

STOP_WORDS_PATH = "DATA/StopWords_Ironia.txt"
AUX_VERBS_PATH  = "DATA/stop-words-vaux.txt"
ALLOWED_CHARS=[' ','-','/']
EMBEDDINGS_PATH = "DATA/embedding_features.pkl"

def clean_word(word):
    return ''.join([c for c in word if (c in ALLOWED_CHARS or c.isalpha())])            


def pairwise_distances(in_file, out_file, features_path, distance="cosine"):    
    """
        Compute the pairwise distances of words in literal titles
    """

    #open embedding matrix    
    with codecs.open(features_path,"r") as fid:
        wrd2idx, E = cPickle.load(fid)
    #open stop words
    with codecs.open(STOP_WORDS_PATH,"r","utf-8") as fid:
        stop_words = fid.read().split()
    #open stop words aux verbs
    with codecs.open(AUX_VERBS_PATH,"r","utf-8") as fid:
        aux_verbs = fid.read().split()        
        #ignore lines starting with #
        aux_verbs = [w for w in aux_verbs if not w.startswith("#")]

    #add aux verbs to the stop word list        
    stop_words += aux_verbs    
    #open titles
    with codecs.open(in_file,"r","utf-8") as fid:        
        titles = fid.read().lower()
    doc_id = 0
    with codecs.open(out_file,"w","utf-8") as fod:                
        for title in titles.split("\n"):
            doc_id+=1
            #remove special chars
            clean_title = clean_word(title)                       
            #remove stop words            
            clean_title = [w.strip() for w in clean_title.split() if w not in stop_words]                        
            #compute pairwise distances
            word_pairs = list(itertools.combinations(clean_title,2))    
            distances = []
            for p in word_pairs:
                w1, w2 = p                                
                #if the words exist in the vocabulary
                if w1 in wrd2idx and w2 in wrd2idx:                    
                    w1_emb = E[:,wrd2idx[w1]]
                    w2_emb = E[:,wrd2idx[w2]]        
                    #distance to the zero vector is not defined                        
                    if np.all(w1_emb==0) or np.all(w2_emb==0):
                        continue
                    else:                        
                        if distance=="cosine":
                            d = cosine(w1_emb,w2_emb)                
                        elif distance=="euclid":
                            d = euclidean(w1_emb,w2_emb)
                        else:                            
                            raise NotImplementedError
                        distances.append(d)
            avg_dist = 0
            if len(distances) > 0: avg_dist = np.mean(distances)
            fod.write(u"#%d\t%s\t%.2f\n"%(doc_id,title,avg_dist))

def pairwise_distances_details(in_file, out_file, features_path, distance="cosine"):    
    """
        Compute the pairwise distances of words in literal titles
    """

    #open embedding matrix    
    with codecs.open(features_path,"r") as fid:
        wrd2idx, E = cPickle.load(fid)
    #open stop words
    with codecs.open(STOP_WORDS_PATH,"r","utf-8") as fid:
        stop_words = fid.read().split()
    #open stop words aux verbs
    with codecs.open(AUX_VERBS_PATH,"r","utf-8") as fid:
        aux_verbs = fid.read().split()        
        #ignore lines starting with #
        aux_verbs = [w for w in aux_verbs if not w.startswith("#")]

    #add aux verbs to the stop word list        
    stop_words += aux_verbs    
    #open titles
    with codecs.open(in_file,"r","utf-8") as fid:        
        titles = fid.read().lower()
    doc_id = 0
    with codecs.open(out_file,"w","utf-8") as fod:                
        for title in titles.split("\n"):
            doc_id+=1
            #remove special chars
            clean_title = clean_word(title)                       
            #remove stop words            
            clean_title = [w.strip() for w in clean_title.split() if w not in stop_words]            
            fod.write(u"#%d\t%s\n"%(doc_id,title))
            #compute pairwise distances
            word_pairs = list(itertools.combinations(clean_title,2))    
            for p in word_pairs:
                w1, w2 = p                                
                #if the words exist in the vocabulary
                if w1 in wrd2idx and w2 in wrd2idx:                    
                    w1_emb = E[:,wrd2idx[w1]]
                    w2_emb = E[:,wrd2idx[w2]]        
                    #distance to the zero vector is not defined                        
                    if np.all(w1_emb==0) or np.all(w2_emb==0):
                        continue
                    else:                        
                        if distance=="cosine":
                            c = cosine(w1_emb,w2_emb)                
                        elif distance=="euclid":
                            c = euclidean(w1_emb,w2_emb)
                        else:
                            print c
                            raise NotImplementedError
                        fod.write("%d\t%s-%s\t%.2f\n" %(doc_id,w1,w2,c))

IN_FILES = ["_semtag_dataset_webanno_tfidf_inimigo.txt","_semtag_dataset_webanno_tfidf_publico.txt" ]
pairwise_distances(IN_FILES[0], "DATA/inimigo_distances.txt", EMBEDDINGS_PATH)
pairwise_distances(IN_FILES[1], "DATA/publico_distances.txt", EMBEDDINGS_PATH)

import codecs
import pickle
# from ipdb import set_trace
import numpy as np

EMBEDDINGS_PATH = "DATA/publico_800.txt"
OUT  = "DATA/embedding_features.pkl"
IN_FILES = ["DATA/_semtag_dataset_webanno_tfidf_inimigo.txt", "DATA/_semtag_dataset_webanno_tfidf_publico.txt" ]

txt = []
for in_file in IN_FILES:
    with codecs.open(in_file,"r","utf-8") as fid:
        txt += fid.readlines()
#words
words = [w for m in txt for w in m.split()]
#unique words
words = list(set(words))
#word index
wrd2idx = {w:i for i,w in enumerate(words)}
#read embedding file
with open(EMBEDDINGS_PATH) as fid:
    voc_size = len(wrd2idx)        
    _, emb_size = fid.readline().split() 
    E = np.zeros((int(emb_size), voc_size)).astype(float)
    for line in fid.readlines():            
        items = line.split()
        wrd   = items[0]#.decode("utf-8")
        if wrd in wrd2idx:
            E[:, wrd2idx[wrd]] = np.array(items[1:]).astype(float)                
# set_trace()
# Number of out of embedding vocabulary embeddings
n_OOEV = np.sum((E.sum(0) == 0).astype(int))
perc = n_OOEV*100./len(wrd2idx)
print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" % (n_OOEV, len(wrd2idx), perc)) 
#save embeddings and word index
with open(OUT,"w") as fo:
    pickle.dump((wrd2idx,E),fo,pickle.HIGHEST_PROTOCOL)
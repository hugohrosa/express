import nltk
from os.path import join
import cPickle

contents = nltk.data.load(join("DATA/pt-mymap.map"), format="text")
tagmap = dict()
tagmap[""] = "X"
for line in contents.splitlines():
    line = line.strip()
    if line == "" : continue
    fine, coarse = line.split("\t")
    tagmap[fine] = coarse
 

def simplify_tag(t):
    if "+" in t: t = t[t.index("+")+1:]
    if "|" in t: t = t[t.index("|")+1:]
    t = t.lower()
    return tagmap[t]

print "Training Tagger"
dataset1 = nltk.corpus.floresta.tagged_sents( )
dataset2 = nltk.corpus.mac_morpho.tagged_sents( ) 
 
train = [ [ (w ,simplify_tag(t)) for (w,t) in sent ] for sent in dataset1 + dataset2 ]
tagger_fast = nltk.TrigramTagger(train, backoff=nltk.BigramTagger(train, backoff=nltk.UnigramTagger(train, backoff=nltk.DefaultTagger('N'))))
print "Done"

with open("Models/tagger.pkl","wb") as fid:
	cPickle.dump(tagger_fast,fid,cPickle.HIGHEST_PROTOCOL)
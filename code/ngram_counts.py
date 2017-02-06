#Simple script to scrap the google results page


from bs4 import BeautifulSoup
import cPickle
from ipdb import set_trace
import requests
import time
import codecs

COUNTS_DIC = "ngram_counts.pkl"


def ngram_counts(word):

	url = "http://corpora.linguistik.uni-erlangen.de/demos/cgi-bin/Web1T5/Web1T5_freq.perl"

	r = requests.get(url,
	                 params={'query':str(word),
	                         "mode":"Search",
	                         "limit":1,
	                         "threshold":100}
	                )
	
	soup = BeautifulSoup(r.text)		
	x = soup.find_all('table')[2].text.replace("\n","").split()
	# set_trace()
	time.sleep(3)
	#sanity check
	if x[1]==word:
		return int(x[0])
	else:
		return 0

def new_counts_dict():	

	"""
		Initialize a count dictionary with words from the
		vocabulary		
	"""

	IN_FILES = ["../_semtag_dataset_webanno_tfidf_inimigo.txt","../_semtag_dataset_webanno_tfidf_publico.txt" ]

	txt = []
	for in_file in IN_FILES:
	    with codecs.open(in_file,"r","utf-8") as fid:
	        txt += fid.readlines()
	#words
	words = [w for m in txt for w in m.split()]
	#unique words
	words = list(set(words))
	#word index
	wrd2idx = {w:-1 for w in words}

	set_trace()
	
	with open(COUNTS_DIC,"w") as fod:
		cPickle.dump(wrd2idx, fod, cPickle.HIGHEST_PROTOCOL)

def get_frequencies():

	with open(COUNTS_DIC,"r") as fid:
		counts_dict = cPickle.load(fid)	
	#filter phrases and words that already have counts
	words = filter(lambda x: x[1]==-1 and len(x[0].split())==1,  
				   list(counts_dict.items()))
	i=0
	for w, _ in words:		
		try:
			c = ngram_counts(w)
			counts_dict[w] = c
			print "%d/%d: %s %d" % (i,len(words),w,c)
		except KeyboardInterrupt:
			#this is a little hack to allow for
			raise EnvironmentError
		except:		
			pass		
		i+=1
		if i%100==0:			
			#save current counts
			with open(COUNTS_DIC,"w") as fod:
				cPickle.dump(counts_dict, fod, cPickle.HIGHEST_PROTOCOL)
	#save the counts
	with open(COUNTS_DIC,"w") as fod:
		cPickle.dump(counts_dict, fod, cPickle.HIGHEST_PROTOCOL)

#create and save new dictionary with all the vocabulary words
new_counts_dict()
#get the frequencies
get_frequencies()

with open(COUNTS_DIC,"r") as fid:
	counts_dict = cPickle.load(fid)
	print counts_dict



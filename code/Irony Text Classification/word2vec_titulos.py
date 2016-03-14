# -*- coding: utf-8 -*-
import nltk
import gensim
import unicodecsv
import string

data = open('../DATA/data_all.csv', 'rU')
reader = unicodecsv.DictReader(data, encoding = 'utf-8', delimiter = '\t')

sentences = []
sent_id=0

exclude = set(unicode(string.punctuation))
exclude.add(u'“')
exclude.add(u'‘')

for row in reader:
    if int(row['num_de_anotadores_total'])>=1:
        if sent_id>4:
            break
        sent_id+=1
        text = ''.join(ch for ch in row['texto'] if ch not in exclude)
        tks = nltk.word_tokenize(text)
        sentences.append(tks)

model = gensim.models.Word2Vec(sentences, min_count=1)
model.save('imparity_model_5')
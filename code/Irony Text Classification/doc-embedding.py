import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten , AutoEncoder
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument

print ("")
print ("Reading pre-trained word embeddings...")
embeddings_dim = 300
embeddings = dict( )
embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True )

print ("Reading the dataset from nlp.stanford.edu/projects/snli/ into memory...")
maxlen = 100
max_features = 10000
trainA = [ row["sentence1"] for row in csv.DictReader(open("snli_1.0/snli_1.0_test.txt") , delimiter='\t') if row["gold_label"] == "entailment" ]
trainB = [ row["sentence2"] for row in csv.DictReader(open("snli_1.0/snli_1.0_test.txt") , delimiter='\t') if row["gold_label"] == "entailment" ]
tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tokenizer.fit_on_texts( trainA + trainB )
trainA = sequence.pad_sequences( tokenizer.texts_to_sequences( trainA ) , maxlen=maxlen )
trainB = sequence.pad_sequences( tokenizer.texts_to_sequences( trainB ) , maxlen=maxlen )
embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
for word,index in tokenizer.word_index.items():
  if index < max_features:
    try: embedding_weights[index,:] = embeddings[word]
    except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )

print ("Train a model to generate phrase embeddings...")
np.random.seed(0)
encoder = Sequential()
encoder.add(Embedding(max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ))
encoder.add(LSTM(output_dim=embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'))
decoder = Sequential()
decoder.add(Embedding(max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ))
decoder.add(LSTM(output_dim=embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'))
model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False, tie_weights=True))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(trainA, trainB, batch_size=16, nb_epoch=10)

print ("Test the generation of a phrase embedding...")
test = sequence.pad_sequences( tokenizer.texts_to_sequences( [ "this is an example sentence" ] ) , maxlen=maxlen )
print model.predict( test )
print encoder.predict( test )

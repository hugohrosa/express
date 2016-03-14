# -*- coding: utf-8 -*-
import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
import miniball
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.kernel_ridge import KernelRidge
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
import unicodecsv

def biggest(ns,s,n):
    ''' determinar o maior de 3 valores: ns - num anotadores 'não sei', s - num anotadores 'sim', n - num anotadores 'não'
    '''
    Max = ns
    out = 'Não Sei'
    if s > Max:
        Max = s
        out = 'Sim' 
    if n > Max:
        Max = n
        out = 'Não'
        if s > n:
            Max = s
            out = 'Sim'
    return out

print ("")
print ("Reading pre-trained word embeddings...")
embeddings_dim = 300
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 
embeddings = Word2Vec.load_word2vec_format( "../DATA/publico_800.txt" , binary=False )

print ("Reading affective dictionary and training regression model for predicting valence, arousal and dominance...")
affective = dict( )
for row in csv.DictReader(open("13428_2011_131_MOESM1_ESM.csv")): affective[ row["EP-Word"].lower() ] = np.array( [ float( row["Val-M"] ) , float( row["Arou-M"] ) , float( row["Dom-M"] ) ] )
train_matrix = [ ]
train_labels = [ ]
for word,scores in affective.items():
  try: 
    train_matrix.append( embeddings[word] )
    train_labels.append( scores )
  except: continue
train_matrix = np.array( train_matrix )
train_labels = np.array( train_labels )
class VADEstimator(BaseEstimator):
  def fit( self, x , y , size=1 ):
    self.model = Sequential()
    self.model.add(Dense( int( embeddings_dim / 2.0 ) , input_dim=embeddings_dim , init='uniform' , activation='tanh'))
    self.model.add(Dense( int( embeddings_dim / 4.0 ) , init='uniform' , activation='tanh'))
    self.model.add(Dense(size , init='uniform' ) )
    self.model.compile(loss='mse', optimizer='rmsprop')
    self.model = KernelRidge( kernel='rbf' )
    self.model.fit( x , y )
  def predict( self, x ): 
    if isinstance( self.model , Sequential ): return self.model.predict( x , verbose=0 )[ 0 ]
    return self.model.predict( x )
def pearsonr( x , y ): return scipy.stats.pearsonr(x,y)[0]
model = VADEstimator( )
scores = sklearn.cross_validation.cross_val_score( model, train_matrix , train_labels[:,0] , cv=10, scoring=sklearn.metrics.make_scorer( pearsonr ) )
print ("Test with 10 fold CV : correlation for valence: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = sklearn.cross_validation.cross_val_score( model, train_matrix , train_labels[:,1] , cv=10, scoring=sklearn.metrics.make_scorer( pearsonr ) )
print ("Test with 10 fold CV : correlation for arousal: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
scores = sklearn.cross_validation.cross_val_score( model, train_matrix , train_labels[:,2] , cv=10, scoring=sklearn.metrics.make_scorer( pearsonr ) )
print ("Test with 10 fold CV : correlation for dominance: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
model.fit( train_matrix , train_labels , size=3 )
print ("")

print ("Reading text data for classification and building representations...")
max_features = 20000
maxlen = 50
#data = [ ( txt, 0 ) for txt in open('twitDB_regular.csv').readlines() ] + [ ( txt, 1 ) for txt in open('twitDB_sarcasm.csv').readlines() ]
data = [ (row["texto"].encode('utf-8').lower(), biggest(int(row['naosei_ironico']), int(row['sim_ironico']), int(row['nao_ironico'])))  for row in unicodecsv.DictReader(open('../DATA/data_all.csv', 'rU') , encoding = 'utf-8', delimiter = '\t') if int(row["num_de_anotadores_total"]) >= 1 ]
random.shuffle( data )
train_size = int(len(data) * 0.8)
#train_texts = [ txt.lower().replace("#sarcasm","").replace("#irony","").replace("#sarcastic","") for ( txt, label ) in data[0:train_size] ]
train_texts = [ txt for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt for ( txt, label ) in data[train_size:-1] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tokenizer.fit_on_texts(train_texts)
train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=maxlen )
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=maxlen )
train_matrix = tokenizer.texts_to_matrix( train_texts )
test_matrix = tokenizer.texts_to_matrix( test_texts )
embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
affective_weights = np.zeros( ( max_features , 3 ) )
for word,index in tokenizer.word_index.items():
  try: 
    if not affective.has_key(word) : affective[word] = np.array( model.predict( np.array( embedding[word] ).reshape(1, -1) )[0] )
  except: affective[word] = np.array( [ 5.0 , 5.0 , 5.0 ] )
  if index < max_features:
    try: 
      embedding_weights[index,:] = embeddings[word]
      affective_weights[index,:] = affective[word]
    except: 
      embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
      affective_weights[index,:] = [ 5.0 , 5.0 , 5.0 ]

print ("Computing features based on semantic volume...")
train_features = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
test_features = np.zeros( ( test_matrix.shape[0] , 1 ) )
for i in range( train_features.shape[0] ):
  aux = [ ]
  for word in train_texts[i].split(" "):
    try: aux.append( embeddings[word] )
    except: continue
  if len( aux ) > 0 : train_features[i,0] = miniball.Miniball( np.array( aux ) ).squared_radius()
for i in range( test_features.shape[0] ):
  aux = [ ]  
  for word in test_texts[i].split(" "): 
    try: aux.append( embeddings[word] )
    except: continue 
  if len( aux ) > 0 : test_features[i,0] = miniball.Miniball( np.array( aux ) ).squared_radius()

print ("Computing features based on affective scores...")
train_features_avg = np.zeros( ( train_matrix.shape[0] , 3 ) ) 
test_features_avg = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_min = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_min = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_max = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_max = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_dif = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_dif = np.zeros( ( test_matrix.shape[0] , 3 ) )
for i in range( train_matrix.shape[0] ):
  aux = [ ]
  for word in train_texts[i].split(" "):
    try: aux.append( affective[word] )
    except: continue
  if len( aux ) > 0 : 
    train_features_avg[i,0] = np.average( np.array( aux ) )
    train_features_min[i,0] = np.min( np.array( aux ) )
    train_features_max[i,0] = np.max( np.array( aux ) )
    train_features_dif[i,0] = np.max( np.array( aux ) ) - np.min( np.array( aux ) )
for i in range( test_matrix.shape[0] ):
  aux = [ ]
  for word in test_texts[i].split(" "):
    try: aux.append( affective[word] )
    except: continue
  if len( aux ) > 0 : 
    test_features_avg[i,0] = np.average( np.array( aux ) )
    test_features_min[i,0] = np.min( np.array( aux ) )    
    test_features_max[i,0] = np.max( np.array( aux ) )
    test_features_dif[i,0] = np.max( np.array( aux ) ) - np.min( np.array( aux ) )
train_features = np.hstack( ( train_features_avg , train_features_min , train_features_max , train_features_dif ) )
test_features = np.hstack( ( test_features_avg , test_features_min, test_features_max, test_features_dif ) )

print ("")
print ("Method = SVM with bag-of-words features")
model = LinearSVC( random_state=0 )
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = SVM with bag-of-words features plus extra features")
model = LinearSVC( random_state=0 )
train_matrix = np.hstack( (train_matrix,train_features) )
test_matrix = np.hstack( (test_matrix,test_features) )
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = MLP with bag-of-words features plus extra features")
np.random.seed(0)
train_matrix = np.hstack( (train_matrix,train_features) )
test_matrix = np.hstack( (test_matrix,test_features) )
model = Sequential()
model.add(Dense(embeddings_dim, input_dim=train_matrix.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(embeddings_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
model.fit( train_matrix , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False)
results = model.predict_classes( test_matrix )
train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = LSTM")
np.random.seed(0)
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ))
model.add(LSTM(output_dim=embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = Bidirectional LSTM")
np.random.seed(0)
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding( max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ), name='embedding', input='input')
model.add_node(LSTM(embeddings_dim), name='forward', input='embedding')
model.add_node(LSTM(embeddings_dim, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')
model.compile('adam', {'output': 'binary_crossentropy'})
model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = CNN-LSTM")
np.random.seed(0)
filter_length = 3
nb_filter = 64
pool_length = 2
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(embeddings_dim))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = SVM with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = LabeledLineSentence( train_texts + test_texts )
model.build_vocab( sentences )
model.train( sentences )
for w in  model.vocab.keys(): 
    try : model[w] = embeddings[w]
    except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = LinearSVC( random_state=0 )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

#print ("Method = MLP with doc2vec features plus additional features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = train_texts + test_texts
sentences = LabeledLineSentence( sentences )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try : model[w] = embeddings[w]
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
train_rep = np.hstack( (train_rep,train_features) )
test_rep = np.hstack( (test_rep,test_features) )
model = Sequential()
model.add(Dense(embeddings_dim, input_dim=train_rep.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(embeddings_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
model.fit( train_rep , train_labels , nb_epoch=1, batch_size=16, show_accuracy=False)
results = model.predict_classes( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = LSTM embeddings plus LSTM affective scores")
np.random.seed(0)
model1 = Sequential()
model1.add(Embedding(max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ))
model1.add(LSTM(output_dim=embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'))
model1.add(Dropout(0.5))
model1.add(Dense(embeddings_dim))
model2 = Sequential()
model2.add(Embedding(max_features, 3, input_length=maxlen, mask_zero=True, weights=[affective_weights] ))
model2.add(LSTM(output_dim=3, activation='sigmoid', inner_activation='hard_sigmoid'))
model2.add(Dropout(0.5))
model1.add(Dense(3))
model = Sequential()
model.add( Merge([model1, model2], mode='concat' ))
model.add(Flatten())
model.add(Dense( int( embeddings_dim / 2.0 ) , activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

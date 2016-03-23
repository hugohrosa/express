# -*- coding: utf-8 -*-
#from ipdb import set_trace
import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
import sys
#import cPickle
import miniball
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import MultinomialNB
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
import codecs
import unicodecsv
from sklearn.cross_validation import KFold

log = codecs.open("log_NEW_irony_detection_test_50-50.txt","w","utf-8")
log = sys.stdout

log.write("\n")
log.write("Reading pre-trained word embeddings...\n")
embeddings_dim = 800
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 
embeddings = Word2Vec.load_word2vec_format( "../DATA/publico_800.txt" , binary=False )

# with open("/ffs/tmp/samir/gensim_loaded.pkl","w") as fd:
#   cPickle.dump(embeddings,fd,-1)
# with open("/ffs/tmp/samir/gensim_loaded.pkl","r") as fd:
#   embeddings = cPickle.load(fd)
log.write("Reading affective dictionary and training regression model for predicting valence, arousal and dominance...\n")
affective = dict( )
#for row in csv.DictReader(open("Ratings_Warriner_et_al.csv")): affective[ row["Word"].lower() ] = np.array( [ float( row["V.Mean.Sum"] ) , float( row["A.Mean.Sum"] ) , float( row["D.Mean.Sum"] ) ] )
for row in csv.DictReader(open("13428_2011_131_MOESM1_ESM.csv")): affective[ row["EP-Word"].lower() ] = np.array( [ float( row["Val-M"] ) , float( row["Arou-M"] ) , float( row["Dom-M"] ) ] )
train_matrix = [ ]
train_labels = [ ]
for word,scores in affective.items():
  try: 
    train_matrix.append( embeddings[word] )
    train_labels.append( scores )
  except: continue
  # remove line below (in order to debug the code faster, I'm limiting the number of words that are used in the regression models... remove when performing tests)
  if len( train_matrix ) > 500 : break
train_matrix = np.array( train_matrix )
train_labels = np.array( train_labels )
# class VADEstimator(BaseEstimator):
#   def fit( self, x , y , size=1 ):
#     self.model = Sequential()
#     self.model.add(Dense( int( embeddings_dim / 2.0 ) , input_dim=embeddings_dim , init='uniform' , activation='tanh'))
#     self.model.add(Dense( int( embeddings_dim / 4.0 ) , init='uniform' , activation='tanh'))
#     self.model.add(Dense(size , init='uniform' ) )
#     self.model.compile(loss='mse', optimizer='rmsprop')
#     self.model = KernelRidge( kernel='poly' , degree=4 )
#     self.model.fit( x , y )
#   def predict( self, x ): 
#     if isinstance( self.model , Sequential ): return self.model.predict( x , verbose=0 )[ 0 ]
#     return self.model.predict( x )
# def pearsonr( x , y ): return scipy.stats.pearsonr(x,y)[0]
# model = VADEstimator( )
# scores = sklearn.cross_validation.cross_val_score( model, train_matrix , train_labels[:,0] , cv=10, scoring=sklearn.metrics.make_scorer( pearsonr ) )
# log.write("Test with 10 fold CV : correlation for valence: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)+"\n")
# scores = sklearn.cross_validation.cross_val_score( model, train_matrix , train_labels[:,1] , cv=10, scoring=sklearn.metrics.make_scorer( pearsonr ) )
# log.write("Test with 10 fold CV : correlation for arousal: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)+"\n") 
# scores = sklearn.cross_validation.cross_val_score( model, train_matrix , train_labels[:,2] , cv=10, scoring=sklearn.metrics.make_scorer( pearsonr ) )
# log.write("Test with 10 fold CV : correlation for dominance: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)+"\n") 
# model.fit( train_matrix , train_labels , size=3 )
# log.write("\n")

# with open("/ffs/tmp/samir/vade.pkl","w") as fd:
#   cPickle.dump(model, fd, -1)

log.write("Reading text data for classification and building representations...\n")
# increase the number of features to 25000 (this corresponds to the number of words in the vocabulary... increase while you have enough memory, and its now set to 20 in order to debug the code faster)
max_features = 20
maxlen = 50
# data = [ ( txt, 0 ) for txt in open('twitDB_regular.csv').readlines() ] + [ ( txt, 1 ) for txt in open('twitDB_sarcasm.csv').readlines() ]
# random.shuffle( data )
# train_size = int(len(data) * 0.8)
# train_texts = [ txt.lower().replace("#sarcasm","").replace("#irony","").replace("#sarcastic","") for ( txt, label ) in data[0:train_size] ]
# test_texts = [ txt.lower().replace("#sarcasm","").replace("#irony","").replace("#sarcastic","") for ( txt, label ) in data[train_size:-1] ]
# train_labels = [ label for ( txt , label ) in data[0:train_size] ]
# test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
lbl_y = 'Imparidade'
lbl_n = 'Não Imparidade'
imp = 0
n_imp = 0
split_trn = 1000
split_tst = 133
# for row in unicodecsv.DictReader(open('../DATA/data_all.csv', 'rU') , encoding = 'utf-8', delimiter = '\t'):
#     if int(row[lbl_y])>=1 and imp < split_trn:
#         #training set for imparity 1000 samples
#         imp+=1
#         train_texts.append(row['texto'].encode('utf-8').lower())
#         train_labels.append(1)
#     elif int(row[lbl_y])==0 and n_imp < split_trn:
#         #training set for not imparity 1000 samples
#         n_imp+=1
#         train_texts.append(row['texto'].encode('utf-8').lower())
#         train_labels.append(0)
#     elif int(row[lbl_y])>=1 and imp >= split_trn  and imp < split_trn + split_tst:
#         #test set for imparity 133 samples
#         imp+=1
#         test_texts.append(row['texto'].encode('utf-8').lower())
#         test_labels.append(1)
#     elif int(row[lbl_y])==0 and n_imp >= split_trn  and n_imp < split_trn + split_tst:
#         #test set for not imparity 133 samples
#         n_imp+=1
#         test_texts.append(row['texto'].encode('utf-8').lower())
#         test_labels.append(0)

maxlen = 0
train = []
test = []
#for row in unicodecsv.DictReader(open('../DATA/data_all.csv', 'rU') , encoding = 'utf-8', delimiter = '\t'):
for row in csv.DictReader(open('../DATA/data_all.csv', 'rU') , delimiter = '\t'):
    if len(row['texto'].split())>maxlen: maxlen = len(row['texto'].split())
    if int(row[lbl_y])>=1 and imp < split_trn:
        #training set for imparity 1000 samples
        imp+=1
        train.append((row['texto'].lower(),1))
        #train_texts.append(row['texto'].encode('utf-8').lower())
        #train_labels.append(1)
    elif int(row[lbl_y])==0 and n_imp < split_trn:
        #training set for not imparity 1000 samples
        n_imp+=1
        train.append((row['texto'].lower(),0))
        #train_texts.append(row['texto'].encode('utf-8').lower())
        #train_labels.append(0)
    elif int(row[lbl_y])>=1 and imp >= split_trn  and imp < split_trn + split_tst:
        #test set for imparity 133 samples
        imp+=1
        test.append((row['texto'].lower(),1))
        #test_texts.append(row['texto'].encode('utf-8').lower())
        #test_labels.append(1)
    elif int(row[lbl_y])==0 and n_imp >= split_trn  and n_imp < split_trn + split_tst:
        #test set for not imparity 133 samples
        n_imp+=1
        test.append((row['texto'].lower(),0))
        #test_texts.append(row['texto'].encode('utf-8').lower())
        #test_labels.append(0)
print('data loaded')

random.shuffle(train)
random.shuffle(test)
train_texts = [ txt for (txt,lbl) in train ]
train_labels = [ lbl for (txt,lbl) in train ]
test_texts = [ txt for (txt,lbl) in test ]
test_labels = [ lbl for (txt, lbl) in test ]

#SILVIO: data should be shuffled! - HUGO: agreed and shuffled
#SILVIO: shouldn't we be using the same tokenizer from the other experiments?
#compute the vocabulary
cc = {w:None for t in train_texts+test_texts for w in t.split()}
max_features = len(cc.keys())
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
    if not affective.has_key(word) : affective[word] = np.array( model.predict( np.array( embeddings[word] ).reshape(1, -1) )[0] )
  except: affective[word] = np.array( [ 5.0 , 5.0 , 5.0 ] )
  if index < max_features:
    try: 
      embedding_weights[index,:] = embeddings[word]
      affective_weights[index,:] = affective[word]
    except: 
      embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
      affective_weights[index,:] = [ 5.0 , 5.0 , 5.0 ]

log.write("Computing features based on semantic volume...\n")
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

log.write("Computing features based on affective scores...\n")
train_features_avg = np.zeros( ( train_matrix.shape[0] , 3 ) ) 
test_features_avg = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_stdev = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_stdev = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_min = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_min = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_max = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_max = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_dif = np.zeros( ( train_matrix.shape[0] , 3 ) )
test_features_dif = np.zeros( ( test_matrix.shape[0] , 3 ) )
train_features_seq = np.zeros( ( train_matrix.shape[0] , 2 ) )
test_features_seq = np.zeros( ( test_matrix.shape[0] , 2 ) )
for i in range( train_matrix.shape[0] ):
  aux = [ ]
  for word in train_texts[i].split(" "):
    try: aux.append( affective[word] )
    except: continue
  if len( aux ) > 0 : 
    train_features_avg[i,0] = np.average( np.array( aux ) )
    train_features_stdev[i,0] = np.std( np.array( aux ) )
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
    test_features_stdev[i,0] = np.std( np.array( aux ) )
    test_features_min[i,0] = np.min( np.array( aux ) )    
    test_features_max[i,0] = np.max( np.array( aux ) )
    test_features_dif[i,0] = np.max( np.array( aux ) ) - np.min( np.array( aux ) )
for i in range( train_matrix.shape[0] ):
  train_features_seq[i] = [0,0]
  prev = -1
  for word in train_texts[i].split(" "):
    try: 
      if( prev != -1 and ( ( prev < 5.0 and affective[word][0] > 5.0 ) or ( prev > 5.0 and affective[word][0] < 5.0 ) ) ): train_features_seq[i][0] += 1.0
      if( prev != -1 and abs( prev - affective[word][0] ) > 3.0 ): train_features_seq[i][1] += 1.0
      prev = affective[word][0]
    except: prev = -1
for i in range( test_matrix.shape[0] ):
  test_features_seq[i] = [0,0]
  prev = -1
  for word in test_texts[i].split(" "):
    try:
      if( prev != -1 and ( ( prev < 5.0 and affective[word][0] > 5.0 ) or ( prev > 5.0 and affective[word][0] < 5.0 ) ) ): test_features_seq[i][0] += 1.0
      if( prev != -1 and abs( prev - affective[word][0] ) > 3.0 ): test_features_seq[i][1] += 1.0 
      prev = affective[word][0]
    except: prev = -1
train_features = np.hstack( ( train_features_avg , train_features_stdev , train_features_min , train_features_max , train_features_dif , train_features_seq ) )
test_features = np.hstack( ( test_features_avg , test_features_stdev, test_features_min, test_features_max, test_features_dif , test_features_seq ) )

log.write("\n")
log.write("Method = Linear SVM with bag-of-words features\n")
#model = LinearSVC( random_state=0 )
#model.fit( train_matrix , train_labels )
#results = model.predict( test_matrix )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = NB-SVM with bag-of-words features\n")
#model = MultinomialNB( fit_prior=False )
#model.fit( train_matrix , train_labels )
#train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
#test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
#model = LinearSVC( random_state=0 )
#model.fit( train_matrix , train_labels )
#results = model.predict( test_matrix )
#train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - model.intercept_.shape[0] ]
#test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - model.intercept_.shape[0] ]
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = Linear SVM with bag-of-words features plus extra features\n")
#train_matrix = np.hstack( (train_matrix,train_features) )
#test_matrix = np.hstack( (test_matrix,test_features) )
#model = LinearSVC( random_state=0 )
#model.fit( train_matrix , train_labels )
#results = model.predict( test_matrix )
#train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
#test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = NB-SVM with bag-of-words features plus extra features\n")
#train_matrix = np.hstack( (train_matrix,train_features) )
#test_matrix = np.hstack( (test_matrix,test_features) )
#model = MultinomialNB( fit_prior=False )
#model.fit( train_matrix , train_labels )
#train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
#test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
#model = LinearSVC( random_state=0 )
#model.fit( train_matrix , train_labels )
#results = model.predict( test_matrix )
#train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] - model.intercept_.shape[0] ]
#test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] - model.intercept_.shape[0] ]
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = MLP with bag-of-words features plus extra features\n")
#np.random.seed(0)
#train_matrix = np.hstack( (train_matrix,train_features) )
#test_matrix = np.hstack( (test_matrix,test_features) )
#model = Sequential()
#model.add(Dense(embeddings_dim, input_dim=train_matrix.shape[1], init='uniform', activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(embeddings_dim, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
#model.fit( train_matrix , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False)
#results = model.predict_classes( test_matrix )
#train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
#test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = Stack of two LSTMs\n")
#np.random.seed(0)
#model = Sequential()
#model.add(Embedding(max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ))
#model.add(Dropout(0.1))
#model.add(LSTM(output_dim=embeddings_dim / 2.0, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
#model.add(Dropout(0.1))
#model.add(LSTM(output_dim=embeddings_dim / 2.0, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
#results = model.predict_classes( test_sequences )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = CNN from the paper 'Convolutional Neural Networks for Sentence Classification'\n")
np.random.seed(0)
nb_filter = 1200
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(max_features, embeddings_dim, input_length=maxlen, mask_zero=False, weights=[embedding_weights] ), name='embedding', input='input')
model.add_node(Dropout(0.), name='dropout_embedding', input='embedding')
for n_gram in [3, 5, 7]:
    model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=maxlen), name='conv_' + str(n_gram), input='dropout_embedding')
    model.add_node(MaxPooling1D(pool_length=maxlen - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
    model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
model.add_node(Dropout(0.), name='dropout', inputs=['flat_' + str(n) for n in [3, 5, 7]])
model.add_node(Dense(1, input_dim=nb_filter * len([3, 5, 7])), name='dense', input='dropout')
model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
model.add_output(name='output', input='sigmoid')
model.compile(loss={'output': 'binary_crossentropy'}, optimizer='rmsprop')
# model.fit({'input': train_sequences, 'output': train_labels}, batch_size=256, nb_epoch=1)
original_weights = model.get_weights()
kf = KFold(n=train_sequences.shape[0],n_folds=10)
j=0
acuracies = []
train_labels = np.array(train_labels)
for train, test in kf:
  model.set_weights(original_weights)
  j+=1
  print ("\nfold: %d" % j)
  # set_trace()
  train_X_slice = train_sequences[train]
  train_Y_slice = train_labels[train]
  test_X_slice  = train_sequences[test]
  test_Y_slice  = train_labels[test]
  model.fit({'input': train_X_slice, 'output': train_Y_slice}, batch_size=16, nb_epoch=5)
  results = model.predict({'input':test_X_slice })['output']
  #binarize the outputsore
  results = np.round(results)
  acuracies.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
  log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_Y_slice , results )  ))

#print "Avg accuracy: %.2f" % np.mean(acuracies)
# log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
# log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = Bidirectional LSTM\n")
#np.random.seed(0)
#model = Graph()
#model.add_input(name='input', input_shape=(maxlen,), dtype=int)
#model.add_node(Embedding( max_features, embeddings_dim, input_length=maxlen, mask_zero=True, weights=[embedding_weights] ), name='embedding', input='input')
#model.add_node(LSTM(embeddings_dim), name='forward', input='embedding')
#model.add_node(LSTM(embeddings_dim, go_backwards=True), name='backward', input='embedding')
#model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
#model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
#model.add_output(name='output', input='sigmoid')
#model.compile('adam', {'output': 'binary_crossentropy'})
#model.fit({'input': train_sequences, 'output': train_labels}, batch_size=16, nb_epoch=5)
#results = model.predict_classes( test_sequences )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = CNN-LSTM\n")
#np.random.seed(0)
#filter_length = 3
#nb_filter = 64
#pool_length = 2
#model = Sequential()
#model.add(Embedding(max_features, embeddings_dim, input_length=maxlen, weights=[embedding_weights]))
#model.add(Dropout(0.25))
#model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))
#model.add(MaxPooling1D(pool_length=pool_length))
#model.add(LSTM(embeddings_dim))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
#model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
#results = model.predict_classes( test_sequences )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results ) ) )
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = Linear SVM with doc2vec features\n")
#np.random.seed(0)
#class LabeledLineSentence(object):
#  def __init__(self, data ): self.data = data
#  def __iter__(self):
#    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["SENTENCE_%s" % uid] )
#model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
#sentences = LabeledLineSentence( train_texts + test_texts )
#model.build_vocab( sentences )
#model.train( sentences )
#for w in model.index2word.values(): model[w] = embeddings[w]
#for epoch in range(10):
#    model.train(sentences)
#    model.alpha -= 0.002
#    model.min_alpha = model.alpha
#train_rep = np.array( [ model["SENTENCE_%s" % i] for i in range( train_matrix.shape[0] ) ] )
#test_rep = np.array( [ model["SENTENCE_%s" % (i + train_matrix.shape[0]) ] for i in range( test_matrix.shape[0] ) ] )
#model = LinearSVC( random_state=0 )
#model.fit( train_rep , train_labels )
#results = model.predict( test_rep )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = Non-linear SVM with doc2vec features\n")
#np.random.seed(0)
#class LabeledLineSentence(object):
#  def __init__(self, data ): self.data = data
#  def __iter__(self):
#    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["SENTENCE_%s" % uid] )
#model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
#sentences = LabeledLineSentence( train_texts + test_texts )
#model.build_vocab( sentences )
#model.train( sentences )
#for w in model.index2word.values(): model[w] = embeddings[w]
#for epoch in range(10):
#    model.train(sentences)
#    model.alpha -= 0.002
#    model.min_alpha = model.alpha
#train_rep = np.array( [ model["SENTENCE_%s" % i] for i in range( train_matrix.shape[0] ) ] )
#test_rep = np.array( [ model["SENTENCE_%s" % (i + train_matrix.shape[0]) ] for i in range( test_matrix.shape[0] ) ] )
#model = SVC( random_state=0 , kernel='rbf' )
#model.fit( train_rep , train_labels )
#results = model.predict( test_rep )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = MLP with doc2vec features plus additional features\n")
#np.random.seed(0)
#class LabeledLineSentence(object):
#  def __init__(self, data ): self.data = data
#  def __iter__(self):
#    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
#model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
#sentences = train_texts + test_texts
#sentences = LabeledLineSentence( sentences )
#model.build_vocab( sentences )
#model.train( sentences )
#for w in model.vocab.keys():
#  try : model[w] = embeddings[w]
#  except : continue
#for epoch in range(10):
#    model.train(sentences)
#    model.alpha -= 0.002
#    model.min_alpha = model.alpha
#train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
#test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
#train_matrix_aux = [ ]
#train_labels_aux = [ ]
#for word,scores in affective.items():
#  try: 
#    train_matrix_aux.append( model[word] )
#    train_labels_aux.append( scores )
#  except: 
#    try:
#      train_matrix.append( embeddings[word] )
#      train_labels.append( scores )
#    except: continue
#train_matrix_aux = np.array( train_matrix_aux )
#train_labels_aux = np.array( train_labels_aux )
#model = VADEstimator( )
#model.fit( train_matrix_aux , train_labels_aux , size=3 )
#train_rep = np.hstack( (train_rep, np.array( model.predict( train_rep ) ) ) )
#test_rep = np.hstack( (test_rep, np.array( model.predict( test_rep ) ) ) )
#train_rep = np.hstack( (train_rep,train_features) )
#test_rep = np.hstack( (test_rep,test_features) )
#model = Sequential()
#model.add(Dense(embeddings_dim, input_dim=train_rep.shape[1], init='uniform', activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(embeddings_dim, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
#model.fit( train_rep , train_labels , nb_epoch=1, batch_size=16, show_accuracy=False)
#results = model.predict_classes( test_rep )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = LSTM with embeddings complemented with affective scores\n")
#np.random.seed(0)
#embedding_weights2 = np.zeros( ( max_features , embeddings_dim + 3 ) )
#for index in range( embedding_weights.shape[0] ):
#  if index < max_features: embedding_weights2[index,:] = np.hstack( ( embedding_weights[index,:] , affective_weights[index,:] ))
#model = Sequential()
#model.add(Embedding(max_features, embeddings_dim+3, input_length=maxlen, mask_zero=True, weights=[embedding_weights2] ))
#model.add(Dropout(0.1))
#model.add(LSTM(output_dim=embeddings_dim / 4.0, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
#model.add(Dropout(0.1))
#model.add(LSTM(output_dim=embeddings_dim / 4.0, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#model.fit( train_sequences , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False )
#results = model.predict_classes( test_sequences )
#log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
#log.write(sklearn.metrics.classification_report( test_labels , results ))

log.write("Method = CNN from the paper 'Convolutional Neural Networks for Sentence Classification' with embeddings complemented with affective scores\n")
np.random.seed(0)
nb_filter = 1200
embedding_weights2 = np.zeros( ( max_features , embeddings_dim + 3 ) )
for index in range( embedding_weights.shape[0] ):
  if index < max_features: embedding_weights2[index,:] = np.hstack( ( embedding_weights[index,:] , affective_weights[index,:] ))
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(max_features, embeddings_dim + 3, input_length=maxlen, mask_zero=False, weights=[embedding_weights2]), name='embedding', input='input')
model.add_node(Dropout(0.1), name='dropout_embedding', input='embedding')
for n_gram in [3, 5, 7]:
    model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim + 3, input_length=maxlen), name='conv_' + str(n_gram), input='dropout_embedding')
    model.add_node(MaxPooling1D(pool_length=maxlen - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
    model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
model.add_node(Dropout(0.1), name='dropout', inputs=['flat_' + str(n) for n in [3, 5, 7]])
model.add_node(Dense(1, input_dim=nb_filter * len([3, 5, 7])), name='dense', input='dropout')
model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
model.add_output(name='output', input='sigmoid')
set_trace()
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=16, nb_epoch=1)
#model.fit({'input': train_sequences, 'output': train_labels}, batch_size=256, nb_epoch=1)
#results = model.predict_classes( test_sequences )
results = model.predict(test_sequences)
log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
log.write(sklearn.metrics.classification_report( test_labels , results ))

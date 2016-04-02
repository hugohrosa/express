# -*- coding: utf-8 -*-
#from ipdb import set_trace
import os
import numpy as np
import csv
import keras
import sklearn
import random
import sys
#import cPickle
import miniball
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense , Dropout
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from gensim.models.word2vec import Word2Vec
import codecs
#from sklearn.cross_validation import KFold

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

try:
    log = codecs.open("_log_all_detection_test.txt","w","utf-8")
    log = sys.stdout
    
    log.write("Reading pre-trained word embeddings...\n")
    embeddings_dim = 800
    embeddings = dict( )
    embeddings = Word2Vec.load_word2vec_format( "DATA/publico_800.txt" , binary=False )
      
    log.write("Reading affective dictionary and training regression model for predicting valence, arousal and dominance...\n")
    affective = dict( )
    #for row in csv.DictReader(open("Ratings_Warriner_et_al.csv")): affective[ row["Word"].lower() ] = np.array( [ float( row["V.Mean.Sum"] ) , float( row["A.Mean.Sum"] ) , float( row["D.Mean.Sum"] ) ] )
    for row in csv.DictReader(open("Irony Text Classification/13428_2011_131_MOESM1_ESM.csv")): affective[ row["EP-Word"].lower() ] = np.array( [ float( row["Val-M"] ) , float( row["Arou-M"] ) , float( row["Dom-M"] ) ] )
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

    log.write("Reading text data for classification and building representations...\n")
    # increase the number of features to 25000 (this corresponds to the number of words in the vocabulary... increase while you have enough memory, and its now set to 20 in order to debug the code faster)
    max_features = 20
    maxlen = 50
#     lbl_size = 0
#     for row in csv.DictReader(open('DATA/data_all.csv', 'rU') , delimiter = '\t'):
#         if int(row['num_de_anotadores_total'])==5:
#             if biggest(int(row['naosei_ironico']),int(row['sim_ironico']),int(row['nao_ironico'])) == 'Sim':
#                 lbl_size+=1
#             if biggest(int(row['naosei_ironico']),int(row['sim_ironico']),int(row['nao_ironico'])) == 'Não':
#                 lbl_size+=1
#    lbl_trn = 0
#    lbl_tst = 0
#    split_trn = round(lbl_size*0.8)
#    split_tst = round(lbl_size*0.2)
    lbl_y_trn = 0
    lbl_n_trn = 0
    lbl_y_tst = 0
    lbl_n_tst = 0
    split_trn = 46
    split_tst = 20
    maxlen = 0
    #train = []
    #test = []
    train_y = []
    train_n = []
    test_y = []
    test_n = []
    for row in csv.DictReader(open('DATA/data_all.csv', 'rU') , delimiter = '\t'):
        if int(row['num_de_anotadores_total'])==5:
            if len(row['texto'].split())>maxlen: maxlen = len(row['texto'].split())
            if biggest(int(row['naosei_ironico']),int(row['sim_ironico']),int(row['nao_ironico'])) == 'Sim' and lbl_y_trn < split_trn:  
                #lbl_trn+=1
                #train.append((row['texto'].lower(),1))
                lbl_y_trn+=1
                train_y.append((row['texto'].lower(),1))
            elif biggest(int(row['naosei_ironico']),int(row['sim_ironico']),int(row['nao_ironico'])) == 'Não' and lbl_n_trn < split_trn:
                #lbl_trn+=1
                #train.append((row['texto'].lower(),0))
                lbl_n_trn+=1
                train_n.append((row['texto'].lower(),0))
            elif biggest(int(row['naosei_ironico']),int(row['sim_ironico']),int(row['nao_ironico'])) == 'Sim' and lbl_y_trn >= split_trn  and lbl_y_tst < split_tst:
                #lbl_tst+=1
                #test.append((row['texto'].lower(),1))
                lbl_y_tst+=1
                test_y.append((row['texto'].lower(),1))
            elif biggest(int(row['naosei_ironico']),int(row['sim_ironico']),int(row['nao_ironico'])) == 'Não' and lbl_n_trn >= split_trn  and lbl_n_tst < split_tst:
                #lbl_tst+=1
                #test.append((row['texto'].lower(),0))
                lbl_n_tst+=1
                test_n.append((row['texto'].lower(),0))
    
    train = train_y + train_n
    test = test_y + test_n
    random.shuffle(train)
    random.shuffle(test)
    train_texts = [ txt for (txt,lbl) in train ]
    train_labels = [ lbl for (txt,lbl) in train ]
    test_texts = [ txt for (txt,lbl) in test ]
    test_labels = [ lbl for (txt, lbl) in test ]
    #log.write('TOTAL SPLIT: train ' + str(lbl_trn) + ' - test ' +str(lbl_tst)+'\n')
    log.write('TRAIN SIZE: ' + str(len(train_texts)) + '\tTEST SIZE: ' + str(len(test_texts)) + '\n')
    log.write('TRAIN SPLIT: ironic ' + str(len([lbl for (txt,lbl) in train if lbl==1])) + ' - not ironic ' +str(len([lbl for (txt,lbl) in train if lbl==0]))+'\n')
    log.write('TEST SPLIT : ironic ' + str(len([lbl for (txt,lbl) in test if lbl==1])) + ' - not ironic ' +str(len([lbl for (txt,lbl) in test if lbl==0]))+'\n')

    cc = {w:None for t in train_texts+test_texts for w in t.split()}
    max_features = len(cc.keys())
    tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
    tokenizer.fit_on_texts(train_texts)
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
    
    log.write("\nMethod = Linear SVM with bag-of-words features\n")
    model = LinearSVC( random_state=0 )
    model.fit( train_matrix , train_labels )
    results = model.predict( test_matrix )
    log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
    log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
    
    log.write('Logistic Regression with bag-of-words features\n')
    model = linear_model.LogisticRegression()
    model.fit(train_matrix , train_labels)
    results = model.predict(test_matrix)
    log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
    log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
    
    log.write("Method = NB-SVM with bag-of-words features\n")
    model = MultinomialNB( fit_prior=False )
    model.fit( train_matrix , train_labels )
    train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
    test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
    model = LinearSVC( random_state=0 )
    model.fit( train_matrix , train_labels )
    results = model.predict( test_matrix )
    train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - model.intercept_.shape[0] ]
    test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - model.intercept_.shape[0] ]
    log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
    log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')

    log.write("Method = Linear SVM with bag-of-words features plus extra features\n")
    train_matrix = np.hstack( (train_matrix,train_features) )
    test_matrix = np.hstack( (test_matrix,test_features) )
    model = LinearSVC( random_state=0 )
    model.fit( train_matrix , train_labels )
    results = model.predict( test_matrix )
    train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
    test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
    log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
    log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
    
    log.write('Logistic Regression with bag-of-words features plus extra features\n')
    train_matrix = np.hstack( (train_matrix,train_features) )
    test_matrix = np.hstack( (test_matrix,test_features) )
    model = linear_model.LogisticRegression()
    model.fit(train_matrix , train_labels)
    results = model.predict(test_matrix)
    train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
    test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
    log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
    log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
    
    log.write("Method = NB-SVM with bag-of-words features plus extra features\n")
    train_matrix = np.hstack( (train_matrix,train_features) )
    test_matrix = np.hstack( (test_matrix,test_features) )
    model = MultinomialNB( fit_prior=False )
    model.fit( train_matrix , train_labels )
    train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
    test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
    model = LinearSVC( random_state=0 )
    model.fit( train_matrix , train_labels )
    results = model.predict( test_matrix )
    train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] - model.intercept_.shape[0] ]
    test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] - model.intercept_.shape[0] ]
    log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
    log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
    
    log.write("Method = MLP with bag-of-words features plus extra features\n")
#     np.random.seed(0)
#     train_matrix = np.hstack( (train_matrix,train_features) )
#     test_matrix = np.hstack( (test_matrix,test_features) )
#     model = Sequential()
#     model.add(Dense(embeddings_dim, input_dim=train_matrix.shape[1], init='uniform', activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(embeddings_dim, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
#     #model.fit( train_matrix , train_labels , nb_epoch=5, batch_size=16, show_accuracy=False)
#     model.fit( train_matrix , train_labels , nb_epoch=5, batch_size=256, show_accuracy=False)
#     results = model.predict_classes( test_matrix )
#     train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
#     test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
#     log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
#     log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    log.write('ERROR ' + str(exc_type) + ',' + str(exc_obj) + ',' +fname+',' + str(exc_tb.tb_lineno)+'\n')
    log.close()
log.close()
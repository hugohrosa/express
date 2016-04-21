# -*- coding: utf-8 -*-
#from ipdb import set_trace
import os
import numpy as np
import csv
import keras
import sklearn
import random
import sys
import miniball
from keras.preprocessing.text import Tokenizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from gensim.models.word2vec import Word2Vec
import codecs
from sklearn.cross_validation import KFold
import pickle
import nltk
import itertools


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

def pmi(a, b, co_occurs):
    #This is overcounting the occurrences because if A co-occurs with B, then B also co-occurs with A 
    # set_trace()
    total_occurs = sum([x['total_occurences'] for x in co_occurs.values() if x is not None])
    
    try:
        #P(a)
        p_a = co_occurs[a]["total_occurences"]*1.0 / total_occurs
        #P(b)
        p_b = co_occurs[b]["total_occurences"]*1.0 / total_occurs
        #Note: the co-occurrence data is indexed by the token found on text
        #whereas the co-occurence data in verbetes is indexed by official_name    
        b_official_name = co_occurs[b]["official_name"]
        #P(a,b)
        p_a_b = co_occurs[a]["verbetes"][b_official_name]*1.0 / total_occurs
    except:
        print('EXCEPT ' +a + ' ' + b +  ' no cooccurences ')
        return -1
    #PMI
    if p_a_b ==0:
        return -1
    
    if p_a == 0:
        return -1
    if p_b == 0:
        return -1
    
    pmi = np.log(p_a_b/(p_a*p_b))
    #Normalized PMI
    npmi = pmi/-np.log(p_a_b)
    
    #print a + ',' + b + '\t' + str(npmi)
    return npmi

def getEntities(sent_dict, text):
    entities_per_title = [s['entities'] for s in sent_dict.values()]
    for ents in entities_per_title:
        i = 0
        for e in ents:
            if e in text:
                i+=1
        if i>0 and len(ents) > 0 and i == len(ents):
            return ents
    return []

try:
    log = codecs.open("_log_all_detection_test.txt","w","utf-8")
    log = sys.stdout
    
    log.write("Reading pre-trained word embeddings...\n")
    embeddings_dim = 800
    embeddings = dict( )
    embeddings = Word2Vec.load_word2vec_format( "DATA/publico_800.txt" , binary=False )
         
    log.write("Reading affective dictionary and training regression model for predicting valence, arousal and dominance...\n")
    affective = dict( )
    for row in csv.DictReader(open("Irony Text Classification/Ratings_Warriner_et_al_translated.csv")): affective[ row["Word"].lower() ] = np.array( [ float( row["V.Mean.Sum"] ) , float( row["A.Mean.Sum"] ) , float( row["D.Mean.Sum"] ) ] )
    #for row in csv.DictReader(open("Irony Text Classification/13428_2011_131_MOESM1_ESM.csv")): affective[ row["EP-Word"].lower() ] = np.array( [ float( row["Val-M"] ) , float( row["Arou-M"] ) , float( row["Dom-M"] ) ] )
    train_matrix = [ ]
    train_labels = [ ]
    for word,scores in affective.items():
        try: 
            train_matrix.append( embeddings[word] )
            train_labels.append( scores )
        except: continue
        # remove line below (in order to debug the code faster, I'm limiting the number of words that are used in the regression models... remove when performing tests)
        #if len( train_matrix ) > 500 : break
    train_matrix = np.array( train_matrix )
    train_labels = np.array( train_labels )

    log.write("Reading text data for classification and building representations...\n")
    # increase the number of features to 25000 (this corresponds to the number of words in the vocabulary... increase while you have enough memory, and its now set to 20 in order to debug the code faster)
    #max_features = 25000
    max_features = 25000
    maxlen = 50
    lbl_size = 0
    data = []
    for row in csv.DictReader(open('DATA/data_all.csv', 'rU') , delimiter = '\t'):
        if row['fonte'] == 'Público':
            data.append((row['texto'].lower(),0))
        elif row['fonte'] == 'Inimigo Público':
            data.append((row['texto'].lower(),1))

    data = data[0:2000]            
    random.shuffle(data)
    train_size = int(len(data) * 0.8)
    train_texts = [ txt for ( txt, label ) in data[0:train_size] ]
    test_texts = [ txt for ( txt, label ) in data[train_size:-1] ]
    train_labels = [ label for ( txt , label ) in data[0:train_size] ]
    test_labels = [ label for ( txt , label ) in data[train_size:-1] ]

    #log.write('TOTAL SPLIT: train ' + str(lbl_trn) + ' - test ' +str(lbl_tst)+'\n')
    log.write('TRAIN SIZE: ' + str(len(train_texts)) + '\tTEST SIZE: ' + str(len(test_texts)) + '\n')
    log.write('TRAIN SPLIT: ironic ' + str(len([lbl for lbl in train_labels if lbl==1])) + ' - not ironic ' +str(len([lbl for lbl in train_labels if lbl==0]))+'\n')
    log.write('TEST SPLIT : ironic ' + str(len([lbl for lbl in test_labels if lbl==1])) + ' - not ironic ' +str(len([lbl for lbl in test_labels if lbl==0]))+'\n')

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
    
    log.write("Computing Sentilex features...\n")
    # total number of potentially positive words, negative words and dif.
    with open('DATA/sentilex.pkl',"rb") as sentilex:
        sent_dict = pickle.load(sentilex)
    train_features_pos = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_neg = np.zeros( ( train_matrix.shape[0] , 1 ) )
    #train_features_sent_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
    test_features_pos = np.zeros( ( test_matrix.shape[0] , 1 ) )
    test_features_neg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    #test_features_sent_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
    for i in range( train_matrix.shape[0] ):
        neg = 0
        pos = 0
        for word in train_texts[i].split(" "):
            if word in sent_dict.keys():
                if sent_dict[word]=='NEG':
                    neg+=1
                if sent_dict[word]=='POS':
                    pos+=1
        if neg > 0 or pos > 0: 
            train_features_pos[i,0] = pos
            train_features_neg[i,0] = neg
    for i in range( test_matrix.shape[0] ):
        neg = 0
        pos = 0
        for word in test_texts[i].split(" "):
            if word in sent_dict.keys():
                if sent_dict[word]=='NEG':
                    neg+=1
                if sent_dict[word]=='POS':
                    pos+=1
        test_features_pos[i,0] = pos
        test_features_neg[i,0] = neg

    log.write("Computing Part-of-Speech Tagger...\n")
    with open('Models/tagger.pkl',"rb") as tagger:
        # num de adjectivos / num de palavras
        # num de substantivos / num de palavras
        # num de verbos / num de palavras
        train_features_adj = np.zeros( ( train_matrix.shape[0] , 1 ) )
        train_features_noun = np.zeros( ( train_matrix.shape[0] , 1 ) )
        train_features_verb = np.zeros( ( train_matrix.shape[0] , 1 ) )
        test_features_adj = np.zeros( ( test_matrix.shape[0] , 1 ) )
        test_features_noun = np.zeros( ( test_matrix.shape[0] , 1 ) )
        test_features_verb = np.zeros( ( test_matrix.shape[0] , 1 ) )
        tagger_fast = pickle.load(tagger)
        for i in range( train_matrix.shape[0] ):
            sent_tagged = tagger_fast.tag(nltk.word_tokenize(train_texts[i]))
            sent_tagged = [w for w in sent_tagged if w[0] not in nltk.corpus.stopwords.words('portuguese')] #unicode(nltk.corpus.stopwords.words('portuguese'))]
            train_features_adj[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'ADJ']) / len(sent_tagged)
            train_features_noun[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'NOUN']) / len(sent_tagged)
            train_features_verb[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'VERB']) / len(sent_tagged)
        for i in range( test_matrix.shape[0] ):
            sent_tagged = tagger_fast.tag(nltk.word_tokenize(test_texts[i]))
            sent_tagged = [w for w in sent_tagged if w[0] not in nltk.corpus.stopwords.words('portuguese')] #unicode(nltk.corpus.stopwords.words('portuguese'))]
            test_features_adj[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'ADJ']) / len(sent_tagged)
            test_features_noun[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'NOUN']) / len(sent_tagged)
            test_features_verb[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'VERB']) / len(sent_tagged)
    
    log.write("Computing PMI features...\n")
    # fazer pmi normalizado e pmi regular para titulo e body
    # avg, min, max, std_dev e dif
    #open co-occurrences dictionary
    train_features_pmi_avg = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
    test_features_pmi_avg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_stdev = np.zeros( ( train_matrix.shape[0] , 1 ) )
    test_features_pmi_stdev = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_min = np.zeros( ( train_matrix.shape[0] , 1 ) )
    test_features_pmi_min = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_max = np.zeros( ( train_matrix.shape[0] , 1 ) )
    test_features_pmi_max = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
    test_features_pmi_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
    
    with open("DATA/_new_cooccurs_sapo_Title.pkl","rb") as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'utf-8'
        co_occurs = u.load()
    with open("DATA/_new_sentences_sapo_Title.pkl","rb") as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'utf-8'
        sent_dict = u.load()        
    for i in range( train_matrix.shape[0] ):
        entities = getEntities(sent_dict,train_texts[i])
        pairwise_pmis = []
        entity_pairs = itertools.combinations(entities, 2)               
        for e_a, e_b in entity_pairs:
            pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
        if len(pairwise_pmis)>0:
            train_features_pmi_avg[i,0] = np.average( pairwise_pmis )
            train_features_pmi_stdev[i,0] = np.std( pairwise_pmis )
            train_features_pmi_min[i,0] = np.min( pairwise_pmis )
            train_features_pmi_max[i,0] = np.max( pairwise_pmis )
    for i in range( test_matrix.shape[0] ):
        entities = getEntities(sent_dict,test_texts[i])
        pairwise_pmis = []
        entity_pairs = itertools.combinations(entities, 2)               
        for e_a, e_b in entity_pairs:
            pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
        if len(pairwise_pmis)>0:    
            test_features_pmi_avg[i,0] = np.average( pairwise_pmis )
            test_features_pmi_stdev[i,0] = np.std( pairwise_pmis )
            test_features_pmi_min[i,0] = np.min( pairwise_pmis )
            test_features_pmi_max[i,0] = np.max( pairwise_pmis )
            
        
    #janelas de tamanho variavel
    # contraste de valence (0 a 9)
    
    # contraste na dimensão temporal
    # relevância de um dado termo num dado momento em contraste, procurando noticias na mesma data
    # excluir stopwords, usar só palavras de tf-idf mais alto
    
    
    train_features = np.hstack( ( train_features_avg , train_features_stdev , train_features_min , 
                                  train_features_max , train_features_dif , train_features_seq ,
                                  train_features_pos, train_features_neg , train_features_adj ,
                                  train_features_noun, train_features_verb, train_features_pmi_avg ,
                                  train_features_pmi_stdev, train_features_pmi_min, train_features_pmi_max) )
    test_features = np.hstack( ( test_features_avg , test_features_stdev, test_features_min,
                                 test_features_max, test_features_dif , test_features_seq ,
                                 test_features_pos, test_features_neg , test_features_adj ,
                                 test_features_noun, test_features_verb, test_features_pmi_avg ,
                                  test_features_pmi_stdev, test_features_pmi_min, test_features_pmi_max) )
    
    kf = KFold(n=len(train_matrix),n_folds=1)
    train_labels = np.array(train_labels)
    
    acc = []
    pre = []
    rec = []
    f1s = []
    log.write("\nMethod = Linear SVM with bag-of-words features\n")
    for train_k, test_k in kf:
        train_X_slice = train_matrix[train_k]
        train_Y_slice = train_labels[train_k]
        test_X_slice  = train_matrix[test_k]
        test_Y_slice  = train_labels[test_k]
    
        model = LinearSVC( random_state=0 )
        #model.fit( train_matrix , train_labels )
        #results = model.predict( test_matrix )
        #log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
        #log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
        model.fit( train_X_slice , train_Y_slice )
        results = model.predict(test_X_slice)
        #log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_Y_slice , results )  )+'\n')
        #log.write(sklearn.metrics.classification_report( test_Y_slice , results )+'\n')
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
    log.write("Avg accuracy: %.2f\n" % np.mean(acc))
    log.write(str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    
    acc = []
    pre = []
    rec = []
    f1s = []
    log.write('Logistic Regression with bag-of-words features\n')
    for train_k, test_k in kf:
        train_X_slice = train_matrix[train_k]
        train_Y_slice = train_labels[train_k]
        test_X_slice  = train_matrix[test_k]
        test_Y_slice  = train_labels[test_k]    
        model = linear_model.LogisticRegression()
#         model.fit(train_matrix , train_labels)
#         results = model.predict(test_matrix)
#         log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
#         log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
        model.fit( train_X_slice , train_Y_slice )
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
    log.write("Avg accuracy: %.2f\n" % np.mean(acc))
    log.write(str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    
    acc = []
    pre = []
    rec = []
    f1s = []
    log.write("Method = NB-SVM with bag-of-words features\n")
    for train_k, test_k in kf:
        train_X_slice = train_matrix[train_k]
        train_Y_slice = train_labels[train_k]
        test_X_slice  = train_matrix[test_k]
        test_Y_slice  = train_labels[test_k]    
        model = MultinomialNB( fit_prior=False )
        #model.fit( train_matrix , train_labels )
        #train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
        #test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
        model.fit( train_X_slice , train_Y_slice )
        train_X_slice = np.hstack( (train_X_slice, model.predict_proba( train_X_slice ) ) )
        test_X_slice = np.hstack( (test_X_slice, model.predict_proba( test_X_slice ) ) )
        model = LinearSVC( random_state=0 )
        #model.fit( train_matrix , train_labels )
        #results = model.predict( test_matrix )
        model.fit( train_X_slice, train_Y_slice)
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
        #train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - model.intercept_.shape[0] ]
        #test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - model.intercept_.shape[0] ]
        #log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
        #log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
    log.write("Avg accuracy: %.2f\n" % np.mean(acc))
    log.write(str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    
    
    train_matrix = np.hstack( (train_matrix,train_features) )
    test_matrix = np.hstack( (test_matrix,test_features) )
    acc = []
    pre = []
    rec = []
    f1s = []
    log.write("Method = Linear SVM with bag-of-words features plus extra features\n")
    for train_k, test_k in kf:
        train_X_slice = train_matrix[train_k]
        train_Y_slice = train_labels[train_k]
        test_X_slice  = train_matrix[test_k]
        test_Y_slice  = train_labels[test_k]
     
        model = LinearSVC( random_state=0 )
#         model.fit( train_matrix , train_labels )
#         results = model.predict( test_matrix )
#         train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
#         test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
#         log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
#         log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
        model.fit( train_X_slice , train_Y_slice )
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
    log.write("Avg accuracy: %.2f\n" % np.mean(acc))
    log.write(str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    
    acc = []
    pre = []
    rec = []
    f1s = []  
    log.write('Logistic Regression with bag-of-words features plus extra features\n')
    for train_k, test_k in kf:
        train_X_slice = train_matrix[train_k]
        train_Y_slice = train_labels[train_k]
        test_X_slice  = train_matrix[test_k]
        test_Y_slice  = train_labels[test_k]
    #train_matrix = np.hstack( (train_matrix,train_features) )
    #test_matrix = np.hstack( (test_matrix,test_features) )
        model = linear_model.LogisticRegression()
#     model.fit(train_matrix , train_labels)
#     results = model.predict(test_matrix)
#     train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] ]
#     test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] ]
#     log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
#     log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
        model.fit( train_X_slice , train_Y_slice )
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
    log.write("Avg accuracy: %.2f\n" % np.mean(acc))
    log.write(str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    
    acc = []
    pre = []
    rec = []
    f1s = []  
    log.write("Method = NB-SVM with bag-of-words features plus extra features\n")
    for train_k, test_k in kf:
        train_X_slice = train_matrix[train_k]
        train_Y_slice = train_labels[train_k]
        test_X_slice  = train_matrix[test_k]
        test_Y_slice  = train_labels[test_k]
#     train_matrix = np.hstack( (train_matrix,train_features) )
#     test_matrix = np.hstack( (test_matrix,test_features) )
        model = MultinomialNB( fit_prior=False )
#     model.fit( train_matrix , train_labels )
#     train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
#     test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
        model.fit( train_X_slice , train_Y_slice )
        train_X_slice = np.hstack( (train_X_slice, model.predict_proba( train_X_slice ) ) )
        test_X_slice = np.hstack( (test_X_slice, model.predict_proba( test_X_slice ) ) )
        model = LinearSVC( random_state=0 )
#         model.fit( train_matrix , train_labels )
#         results = model.predict( test_matrix )
#         train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - train_features.shape[1] - model.intercept_.shape[0] ]
#         test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - test_features.shape[1] - model.intercept_.shape[0] ]
#         log.write("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  )+'\n')
#         log.write(sklearn.metrics.classification_report( test_labels , results )+'\n')
        model.fit( train_X_slice, train_Y_slice)
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
    log.write("Avg accuracy: %.2f\n" % np.mean(acc))
    log.write(str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    log.write('ERROR ' + str(exc_type) + ',' + str(exc_obj) + ',' +fname+',' + str(exc_tb.tb_lineno)+'\n')
    log.close()
log.close()
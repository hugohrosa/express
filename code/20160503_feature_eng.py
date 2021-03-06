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
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from gensim.models.word2vec import Word2Vec
import codecs
from sklearn.cross_validation import KFold
import pickle
import nltk
import itertools
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

def kfolds(n_folds,n_elements,val_set=False,shuffle=False):        
    if val_set:
        assert n_folds>2

    X = np.arange(n_elements)
    if shuffle: random.shuffle(X)    
    X = X.tolist()
    slice_size = n_elements // n_folds
    slices =  [X[j*slice_size:(j+1)*slice_size] for j in range(n_folds)]
    #append the remaining elements to the last slice
    slices[-1] += X[n_folds*slice_size:]
    kf = []
    for i in range(len(slices)):
        train = slices[:]        
        test = train.pop(i)
        if val_set:
            try:
                val = train.pop(i)
            except IndexError:
                val = train.pop(-1)                
            #flatten the list of lists
            train = [item for sublist in train for item in sublist]
            kf.append([train,test,val])
        else:

            train = [item for sublist in train for item in sublist]
            kf.append([train,test])
    return kf

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

ALLOWED_CHARS=[' ','-','/']
EMBEDDINGS_PATH = "DATA/embedding_features.pkl"
def clean_word(word):
    return ''.join([c for c in word if (c in ALLOWED_CHARS or c.isalpha())])   

def pairwise_distances(title, wrd2idx, stop_words, distance="cosine"):    
    """
        Compute the pairwise distances of words in literal titles
    """
    #remove special chars
    clean_title = clean_word(title)                       
    #remove stop words            
    clean_title = [w.strip() for w in clean_title.split() if w not in stop_words]                        
    #compute pairwise distances
    word_pairs = list(itertools.combinations(clean_title,2))    
    distances = []
    avg = 0
    std_dev = 0
    min_p = 0
    max_p = 0
    dif = 0
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
            if len(distances) > 0: 
                avg = np.mean(distances)
                std_dev = np.std(distances)
                min_p = np.min(distances)
                max_p = np.max(distances)
                dif = max_p - min_p
    return avg, std_dev, min_p, max_p, dif

def classify(sets, train_matrix, message, coef=False, num_feat = 1):
    c_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    
    log.write("\nMethod = Linear SVM " + message + "\n")
    acc = []
    pre = []
    rec = []
    f1s = []
    for set in sets:
        train_set = set[0]
        validation_set = set[2]
        test_set = set[1]
        
        best_c = 1.0
        best_f1 = 0.0
        for c in c_range:
            model = LinearSVC( random_state=0, C = c)
            model.fit( train_matrix[train_set] , train_labels[train_set] )
            results = model.predict(train_matrix[validation_set])
            f1 = sklearn.metrics.f1_score( train_labels[validation_set], results )
            if  f1 > best_f1:
                best_c = c
                best_f1 = f1
        log.write('BEST C ' + str(best_c)+'\tBEST F1 ' + str(best_f1)+'\n')       
                
        train_90 = train_set + validation_set
        train_X_slice = train_matrix[train_90]
        train_Y_slice = train_labels[train_90]
        test_X_slice = train_matrix[test_set]
        test_Y_slice = train_labels[test_set]    
    
        model = LinearSVC( random_state=0, C = best_c)
        model.fit( train_X_slice , train_Y_slice )
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
        coef.append(model.coef_[0][-num_feat:]) 
    log.write("Overall avg accuracy: %.2f\n" % np.mean(acc))
    log.write("Overall results " + str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    if coef and num_feat > 1:
        log.write("Overall Feature Coef:\n"+str(coef)+'\n\n')
        
    log.write('Logistic Regression ' + message + '\n')
    acc = []
    pre = []
    rec = []
    f1s = []
    for set in sets:
        train_set = set[0]
        validation_set = set[2]
        test_set = set[1]
        
        best_c = 1.0
        best_f1 = 0.0
        for c in c_range:
            model = linear_model.LogisticRegression(C = c)
            model.fit( train_matrix[train_set] , train_labels[train_set] )
            results = model.predict(train_matrix[validation_set])
            f1 = sklearn.metrics.f1_score( train_labels[validation_set], results )
            if f1  > best_f1:
                best_c = c
                best_f1 = f1
        log.write('BEST C ' + str(best_c)+'\tBEST F1 ' + str(best_f1)+'\n')       
                
        train_90 = train_set + validation_set
        train_X_slice = train_matrix[train_90]
        train_Y_slice = train_labels[train_90]
        test_X_slice = train_matrix[test_set]
        test_Y_slice = train_labels[test_set]    
    
        model = model = linear_model.LogisticRegression(C = best_c)
        model.fit( train_X_slice , train_Y_slice )
        results = model.predict(test_X_slice)
        acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
        pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
        rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
        f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
        coef.append(model.coef_[0][-num_feat:]) 
    log.write("Overall avg accuracy: %.2f\n" % np.mean(acc))
    log.write("Overall results " + str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
    if coef and num_feat > 1:
        log.write("Overall Feature Coef:\n"+str(coef)+'\n\n')
        

try:
    log = codecs.open("_log_feature_eng_featureremoval.txt","w","utf-8")
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

    data = data[0:500]  
    #data = data[0:2000]          
    random.shuffle(data)
    train_size = int(len(data) * 0.8)
    train_texts = [ txt for ( txt, label ) in data[0:len(data)-1] ]
#    test_texts = [ txt for ( txt, label ) in data[train_size:-1] ]
    train_labels = [ label for ( txt , label ) in data[0:len(data)-1] ]
#    test_labels = [ label for ( txt , label ) in data[train_size:-1] ]

#    cc = {w:None for t in train_texts+test_texts for w in t.split()}
    cc = {w:None for t in train_texts for w in t.split()}
    max_features = len(cc.keys())
    tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
    tokenizer.fit_on_texts(train_texts)
    train_matrix = tokenizer.texts_to_matrix( train_texts )
#    test_matrix = tokenizer.texts_to_matrix( test_texts )
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
#    test_features = np.zeros( ( test_matrix.shape[0] , 1 ) )
    for i in range( train_features.shape[0] ):
        aux = [ ]
        for word in train_texts[i].split(" "):
            try: aux.append( embeddings[word] )
            except: continue
        if len( aux ) > 0 : train_features[i,0] = miniball.Miniball( np.array( aux ) ).squared_radius()
#         for i in range( test_features.shape[0] ):
#             aux = [ ]  
#         for word in test_texts[i].split(" "): 
#             try: aux.append( embeddings[word] )
#             except: continue 
#         if len( aux ) > 0 : test_features[i,0] = miniball.Miniball( np.array( aux ) ).squared_radius()
         
    log.write("Computing features based on affective scores...\n")
    train_features_avg = np.zeros( ( train_matrix.shape[0] , 3 ) ) 
#    test_features_avg = np.zeros( ( test_matrix.shape[0] , 3 ) )
    train_features_stdev = np.zeros( ( train_matrix.shape[0] , 3 ) )
#    test_features_stdev = np.zeros( ( test_matrix.shape[0] , 3 ) )
    train_features_min = np.zeros( ( train_matrix.shape[0] , 3 ) )
#    test_features_min = np.zeros( ( test_matrix.shape[0] , 3 ) )
    train_features_max = np.zeros( ( train_matrix.shape[0] , 3 ) )
#    test_features_max = np.zeros( ( test_matrix.shape[0] , 3 ) )
    train_features_dif = np.zeros( ( train_matrix.shape[0] , 3 ) )
#    test_features_dif = np.zeros( ( test_matrix.shape[0] , 3 ) )
    train_features_seq = np.zeros( ( train_matrix.shape[0] , 2 ) )
#    test_features_seq = np.zeros( ( test_matrix.shape[0] , 2 ) )
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
#     for i in range( test_matrix.shape[0] ):
#         aux = [ ]
#         for word in test_texts[i].split(" "):
#             try: aux.append( affective[word] )
#             except: continue
#         if len( aux ) > 0 : 
#             test_features_avg[i,0] = np.average( np.array( aux ) )
#             test_features_stdev[i,0] = np.std( np.array( aux ) )
#             test_features_min[i,0] = np.min( np.array( aux ) )    
#             test_features_max[i,0] = np.max( np.array( aux ) )
#             test_features_dif[i,0] = np.max( np.array( aux ) ) - np.min( np.array( aux ) )
    for i in range( train_matrix.shape[0] ):
        train_features_seq[i] = [0,0]
        prev = -1
        for word in train_texts[i].split(" "):
            try: 
                if( prev != -1 and ( ( prev < 5.0 and affective[word][0] > 5.0 ) or ( prev > 5.0 and affective[word][0] < 5.0 ) ) ): train_features_seq[i][0] += 1.0
                if( prev != -1 and abs( prev - affective[word][0] ) > 3.0 ): train_features_seq[i][1] += 1.0
                prev = affective[word][0]
            except: prev = -1
#     for i in range( test_matrix.shape[0] ):
#         test_features_seq[i] = [0,0]
#         prev = -1
#         for word in test_texts[i].split(" "):
#             try:
#                 if( prev != -1 and ( ( prev < 5.0 and affective[word][0] > 5.0 ) or ( prev > 5.0 and affective[word][0] < 5.0 ) ) ): test_features_seq[i][0] += 1.0
#                 if( prev != -1 and abs( prev - affective[word][0] ) > 3.0 ): test_features_seq[i][1] += 1.0 
#                 prev = affective[word][0]
#             except: prev = -1
      
    log.write("Computing Sentilex features...\n")
    # total number of potentially positive words, negative words and dif.
    with open('DATA/sentilex.pkl',"rb") as sentilex:
        sent_dict = pickle.load(sentilex)
    train_features_pos = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_neg = np.zeros( ( train_matrix.shape[0] , 1 ) )
    #train_features_sent_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pos = np.zeros( ( test_matrix.shape[0] , 1 ) )
#    test_features_neg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    #test_features_sent_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
    for i in range( train_matrix.shape[0] ):
        neg = 0
        pos = 0
        w = 0
        for word in train_texts[i].split(" "):
            w+=1
            if word in sent_dict.keys():
                if sent_dict[word]=='NEG':
                    neg+=1
                if sent_dict[word]=='POS':
                    pos+=1
        train_features_pos[i,0] = pos / w
        train_features_neg[i,0] = neg / w
#     for i in range( test_matrix.shape[0] ):
#         neg = 0
#         pos = 0
#         w = 0
#         for word in test_texts[i].split(" "):
#             w+=1
#             if word in sent_dict.keys():
#                 if sent_dict[word]=='NEG':
#                     neg+=1
#                 if sent_dict[word]=='POS':
#                     pos+=1
#         test_features_pos[i,0] = pos / w
#         test_features_neg[i,0] = neg / w
 
    #janelas de tamanho variavel
    # contraste de valence (0 a 9)
    log.write("Computing Arousal Contrast Sliding Window features ...\n")
    val_dif = 2.5
    train_features_ac_w1 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_ac_w2 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_ac_w3 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_ac_w4 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_ac_w5 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    for i in range( train_matrix.shape[0] ):
        arousals = []
        for word in train_texts[i].split(" "):
            if word in affective:
                arousals.append(affective[word][1]) #arousal is index 1
            else:
                arousals.append(0)
        for x in [arousals[i:i+2] for i in range(len(arousals)-1)]: 
            if x[0] > 0 and x[1] > 0 and abs(x[0]-x[1]) > val_dif:
                train_features_ac_w1[i,0] +=1
        for x in [arousals[i:i+3] for i in range(len(arousals)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_ac_w2[i,0] +=1 /len(x)
        for x in [arousals[i:i+4] for i in range(len(arousals)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_ac_w3[i,0] +=1 / len(x)
        for x in [arousals[i:i+5] for i in range(len(arousals)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_ac_w4[i,0] +=1 / len(x)
        for x in [arousals[i:i+6] for i in range(len(arousals)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_ac_w5[i,0] +=1 / len(x)
#     test_features_ac_w1 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_ac_w2 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_ac_w3 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_ac_w4 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_ac_w5 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     for i in range( test_matrix.shape[0] ):
#         arousals = []
#         for word in test_texts[i].split(" "):
#             if word in affective:
#                 arousals.append(affective[word][1]) #arousal is index 1
#             else:
#                 arousals.append(0)
#         for x in [arousals[i:i+2] for i in range(len(arousals)-1)]: 
#             if x[0] > 0 and x[1] > 0 and abs(x[0]-x[1]) > val_dif:
#                 test_features_ac_w1[i,0] +=1
#         for x in [arousals[i:i+3] for i in range(len(arousals)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_ac_w2[i,0] +=1 /len(x)
#         for x in [arousals[i:i+4] for i in range(len(arousals)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_ac_w3[i,0] +=1 / len(x)
#         for x in [arousals[i:i+5] for i in range(len(arousals)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_ac_w4[i,0] +=1 / len(x)
#         for x in [arousals[i:i+6] for i in range(len(arousals)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_ac_w5[i,0] +=1 / len(x)
                   
    #janelas de tamanho variavel
    # contraste de valence (0 a 9)
    log.write("Computing Valence Contrast Sliding Window features ...\n")
    val_dif = 4
    train_features_vc_w1 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_vc_w2 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_vc_w3 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_vc_w4 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    train_features_vc_w5 = np.zeros( ( train_matrix.shape[0] , 1 ) )
    for i in range( train_matrix.shape[0] ):
        valences = []
        for word in train_texts[i].split(" "):
            if word in affective:
                valences.append(affective[word][0]) #valence is index 0
            else:
                valences.append(0)
        for x in [valences[i:i+2] for i in range(len(valences)-1)]: 
            if x[0] > 0 and x[1] > 0 and abs(x[0]-x[1]) > val_dif:
                train_features_vc_w1[i,0] +=1
        for x in [valences[i:i+3] for i in range(len(valences)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_vc_w2[i,0] +=1 /len(x)
        for x in [valences[i:i+4] for i in range(len(valences)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_vc_w3[i,0] +=1 / len(x)
        for x in [valences[i:i+5] for i in range(len(valences)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_vc_w4[i,0] +=1 / len(x)
        for x in [valences[i:i+6] for i in range(len(valences)-1)]: 
            val_interest = [i for i in x if i>0]
            if len(val_interest)>=2:
                for a,b in itertools.combinations(val_interest, 2):
                    if abs(a-b)>val_dif:
                        train_features_vc_w5[i,0] +=1 / len(x)
      
#     test_features_vc_w1 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_vc_w2 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_vc_w3 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_vc_w4 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     test_features_vc_w5 = np.zeros( ( test_matrix.shape[0] , 1 ) )
#     for i in range( test_matrix.shape[0] ):
#         valences = []
#         for word in test_texts[i].split(" "):
#             if word in affective:
#                 valences.append(affective[word][0])
#             else:
#                 valences.append(0)
#         for x in [valences[i:i+2] for i in range(len(valences)-1)]: 
#             if x[0] > 0 and x[1] > 0 and abs(x[0]-x[1]) > val_dif:
#                 test_features_vc_w1[i,0] +=1
#         for x in [valences[i:i+3] for i in range(len(valences)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_vc_w2[i,0] +=1 /len(x)
#         for x in [valences[i:i+4] for i in range(len(valences)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_vc_w3[i,0] +=1 / len(x)
#         for x in [valences[i:i+5] for i in range(len(valences)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_vc_w4[i,0] +=1 / len(x)
#         for x in [valences[i:i+6] for i in range(len(valences)-1)]: 
#             val_interest = [i for i in x if i>0]
#             if len(val_interest)>=2:
#                 for a,b in itertools.combinations(val_interest, 2):
#                     if abs(a-b)>val_dif:
#                         test_features_vc_w5[i,0] +=1 / len(x)
  
    log.write("Computing Part-of-Speech Tagger...\n")
    with open('Models/tagger.pkl',"rb") as tagger:
        train_features_adj = np.zeros( ( train_matrix.shape[0] , 1 ) )
        train_features_noun = np.zeros( ( train_matrix.shape[0] , 1 ) )
        train_features_verb = np.zeros( ( train_matrix.shape[0] , 1 ) )
#         test_features_adj = np.zeros( ( test_matrix.shape[0] , 1 ) )
#         test_features_noun = np.zeros( ( test_matrix.shape[0] , 1 ) )
#         test_features_verb = np.zeros( ( test_matrix.shape[0] , 1 ) )
        tagger_fast = pickle.load(tagger)
        for i in range( train_matrix.shape[0] ):
            sent_tagged = tagger_fast.tag(nltk.word_tokenize(train_texts[i]))
            sent_tagged = [w for w in sent_tagged if w[0] not in nltk.corpus.stopwords.words('portuguese')] #unicode(nltk.corpus.stopwords.words('portuguese'))]
            if len(sent_tagged)>0:
                train_features_adj[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'ADJ']) / len(sent_tagged)
                train_features_noun[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'NOUN']) / len(sent_tagged)
                train_features_verb[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'VERB']) / len(sent_tagged)
#         for i in range( test_matrix.shape[0] ):
#             sent_tagged = tagger_fast.tag(nltk.word_tokenize(test_texts[i]))
#             sent_tagged = [w for w in sent_tagged if w[0] not in nltk.corpus.stopwords.words('portuguese')] #unicode(nltk.corpus.stopwords.words('portuguese'))]
#             if len(sent_tagged)>0:
#                 test_features_adj[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'ADJ']) / len(sent_tagged)
#                 test_features_noun[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'NOUN']) / len(sent_tagged)
#                 test_features_verb[i,0] = len([tag for (word,tag) in sent_tagged if tag == 'VERB']) / len(sent_tagged)
              
    log.write("Computing Pairwise Distances...\n")
    with open("DATA/embedding_features.pkl","rb") as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'ISO-8859-1'
        wrd2idx, E = u.load()
    with open("DATA/StopWords_Ironia.txt","rb") as fid:
        stop_words = fid.read().split()
    train_features_dist_avg = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
#    test_features_dist_avg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_dist_stdev = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_dist_stdev = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_dist_min = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_dist_min = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_dist_max = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_dist_max = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_dist_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_dist_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
    for i in range( train_matrix.shape[0]):
        train_features_dist_avg[i,0], train_features_dist_stdev[i,0], train_features_dist_min[i,0], train_features_dist_max[i,0], train_features_dist_dif[i,0] = pairwise_distances(train_texts[i], wrd2idx, stop_words)
#     for i in range( test_matrix.shape[0]):
#         test_features_dist_avg[i,0], test_features_dist_stdev[i,0], test_features_dist_min[i,0], test_features_dist_max[i,0], test_features_dist_dif[i,0] = pairwise_distances(test_texts[i], wrd2idx, stop_words)
#   
      
    log.write("Computing PMI Title features...\n")
    # fazer pmi normalizado e pmi regular para titulo e body
    train_features_pmi_avg = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
#    test_features_pmi_avg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_stdev = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmi_stdev = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_min = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmi_min = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_max = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmi_max = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmi_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmi_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
      
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
#     for i in range( test_matrix.shape[0] ):
#         entities = getEntities(sent_dict,test_texts[i])
#         pairwise_pmis = []
#         entity_pairs = itertools.combinations(entities, 2)               
#         for e_a, e_b in entity_pairs:
#             pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
#         if len(pairwise_pmis)>0:    
#             test_features_pmi_avg[i,0] = np.average( pairwise_pmis )
#             test_features_pmi_stdev[i,0] = np.std( pairwise_pmis )
#             test_features_pmi_min[i,0] = np.min( pairwise_pmis )
#             test_features_pmi_max[i,0] = np.max( pairwise_pmis )
              
    log.write("Computing PMI body features...\n")
    # fazer pmi normalizado e pmi regular para titulo e body
    train_features_pmibody_avg = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
#    test_features_pmibody_avg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmibody_stdev = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmibody_stdev = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmibody_min = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmibody_min = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmibody_max = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmibody_max = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pmibody_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pmibody_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
      
    with open("DATA/_new_cooccurs_sapo_Body.pkl","rb") as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'utf-8'
        co_occurs = u.load()
    with open("DATA/_new_sentences_sapo_Body.pkl","rb") as fid:
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
            train_features_pmibody_avg[i,0] = np.average( pairwise_pmis )
            train_features_pmibody_stdev[i,0] = np.std( pairwise_pmis )
            train_features_pmibody_min[i,0] = np.min( pairwise_pmis )
            train_features_pmibody_max[i,0] = np.max( pairwise_pmis )
#     for i in range( test_matrix.shape[0] ):
#         entities = getEntities(sent_dict,test_texts[i])
#         pairwise_pmis = []
#         entity_pairs = itertools.combinations(entities, 2)               
#         for e_a, e_b in entity_pairs:
#             pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
#         if len(pairwise_pmis)>0:    
#             test_features_pmibody_avg[i,0] = np.average( pairwise_pmis )
#             test_features_pmibody_stdev[i,0] = np.std( pairwise_pmis )
#             test_features_pmibody_min[i,0] = np.min( pairwise_pmis )
#             test_features_pmibody_max[i,0] = np.max( pairwise_pmis )            
     
    # features de PMI só de entidades
    log.write("Computing PMI NER Verbetes Body features...\n")
    # fazer pmi normalizado e pmi regular para titulo e body
    train_features_pminer_avg = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
#    test_features_pminer_avg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_stdev = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_stdev = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_min = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_min = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_max = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_max = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
      
    with open("DATA/_new_cooccurs_sapo_entities.pkl","rb") as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'utf-8'
        co_occurs = u.load()
    with open("DATA/_new_sentences_sapo_entities.pkl","rb") as fid:
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
            train_features_pminer_avg[i,0] = np.average( pairwise_pmis )
            train_features_pminer_stdev[i,0] = np.std( pairwise_pmis )
            train_features_pminer_min[i,0] = np.min( pairwise_pmis )
            train_features_pminer_max[i,0] = np.max( pairwise_pmis )
#     for i in range( test_matrix.shape[0] ):
#         entities = getEntities(sent_dict,test_texts[i])
#         pairwise_pmis = []
#         entity_pairs = itertools.combinations(entities, 2)               
#         for e_a, e_b in entity_pairs:
#             pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
#         if len(pairwise_pmis)>0:    
#             test_features_pminer_avg[i,0] = np.average( pairwise_pmis )
#             test_features_pminer_stdev[i,0] = np.std( pairwise_pmis )
#             test_features_pminer_min[i,0] = np.min( pairwise_pmis )
#             test_features_pminer_max[i,0] = np.max( pairwise_pmis )
     
    # features de PMI só de entidades
    log.write("Computing PMI NER Verbetes Title features...\n")
    # fazer pmi normalizado e pmi regular para titulo e body
    train_features_pminer_title_avg = np.zeros( ( train_matrix.shape[0] , 1 ) ) 
#    test_features_pminer_title_avg = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_title_stdev = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_title_stdev = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_title_min = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_title_min = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_title_max = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_title_max = np.zeros( ( test_matrix.shape[0] , 1 ) )
    train_features_pminer_title_dif = np.zeros( ( train_matrix.shape[0] , 1 ) )
#    test_features_pminer_title_dif = np.zeros( ( test_matrix.shape[0] , 1 ) )
      
    with open("DATA/_new_cooccurs_sapo_entities_Title.pkl","rb") as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'utf-8'
        co_occurs = u.load()
    with open("DATA/_new_sentences_sapo_entities_Title.pkl","rb") as fid:
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
            train_features_pminer_title_avg[i,0] = np.average( pairwise_pmis )
            train_features_pminer_title_stdev[i,0] = np.std( pairwise_pmis )
            train_features_pminer_title_min[i,0] = np.min( pairwise_pmis )
            train_features_pminer_title_max[i,0] = np.max( pairwise_pmis )
#     for i in range( test_matrix.shape[0] ):
#         entities = getEntities(sent_dict,test_texts[i])
#         pairwise_pmis = []
#         entity_pairs = itertools.combinations(entities, 2)               
#         for e_a, e_b in entity_pairs:
#             pairwise_pmis.append(pmi(e_a, e_b, co_occurs))
#         if len(pairwise_pmis)>0:    
#             test_features_pminer_title_avg[i,0] = np.average( pairwise_pmis )
#             test_features_pminer_title_stdev[i,0] = np.std( pairwise_pmis )
#             test_features_pminer_title_min[i,0] = np.min( pairwise_pmis )
#             test_features_pminer_title_max[i,0] = np.max( pairwise_pmis )           
         
 
    # contraste na dimensão temporal
    # relevância de um dado termo num dado momento em contraste, procurando noticias na mesma data
    # excluir stopwords, usar só palavras de tf-idf mais alto
     
    train_features = np.hstack( ( train_features_avg , train_features_stdev , train_features_min , 
                                  train_features_max , train_features_dif , train_features_seq ,
                                  train_features_pos, train_features_neg , train_features_adj ,
                                  train_features_noun, train_features_verb, train_features_pmi_avg ,
                                  train_features_pmi_stdev, train_features_pmi_min, train_features_pmi_max , 
                                  train_features_pmibody_avg , train_features_pmibody_stdev, train_features_pmibody_min , 
                                  train_features_pmibody_max,  train_features_pminer_avg , train_features_pminer_stdev ,
                                  train_features_pminer_min , train_features_pminer_max, train_features_pminer_title_avg ,
                                  train_features_pminer_title_stdev , train_features_pminer_title_min , train_features_pminer_title_max, 
                                  train_features_dist_avg , train_features_dist_stdev, train_features_dist_min, train_features_dist_max, 
                                  train_features_dist_dif, train_features_vc_w1, train_features_vc_w2, train_features_vc_w3, 
                                  train_features_vc_w4, train_features_vc_w5, train_features_ac_w1, 
                                  train_features_ac_w2, train_features_ac_w3, train_features_ac_w4,
                                  train_features_ac_w5) )
     
#     test_features = np.hstack( ( test_features_avg , test_features_stdev , test_features_min , 
#                                   test_features_max , test_features_dif , test_features_seq ,
#                                   test_features_pos, test_features_neg , test_features_adj ,
#                                   test_features_noun, test_features_verb, test_features_pmi_avg ,
#                                   test_features_pmi_stdev, test_features_pmi_min, test_features_pmi_max , 
#                                   test_features_pmibody_avg , test_features_pmibody_stdev, test_features_pmibody_min , 
#                                   test_features_pmibody_max,  test_features_pminer_avg , test_features_pminer_stdev ,
#                                   test_features_pminer_title_avg , test_features_pminer_title_stdev ,
#                                   test_features_pminer_title_min , test_features_pminer_title_max,
#                                   test_features_pminer_title_min , test_features_pminer_title_max, test_features_dist_avg , 
#                                   test_features_dist_stdev, test_features_dist_min, test_features_dist_max, 
#                                   test_features_dist_dif, test_features_vc_w1, test_features_vc_w2, test_features_vc_w3, 
#                                   test_features_vc_w4, test_features_vc_w5, test_features_ac_w1, 
#                                   test_features_ac_w2, test_features_ac_w3, test_features_ac_w4,
#                                   test_features_ac_w5) )
    
    train_labels = np.array(train_labels)
    #test_labels = np.array(test_labels)
    
    sets = kfolds(10,len(train_labels),val_set=True,shuffle=False)
    c_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    
    classify(sets, train_matrix, "with bag-of-words features")
    
    train_matrix = np.hstack( (train_matrix,train_features) )
    classify(sets, train_matrix, "with EXTRA features" , coef=True, num_feat = train_features.shape[1] )
    
    removed = np.delete(train_matrix, [train_features_vc_w1, train_features_vc_w2, train_features_vc_w3, 
                                  train_features_vc_w4, train_features_vc_w5, train_features_ac_w1, 
                                  train_features_ac_w2, train_features_ac_w3, train_features_ac_w4,
                                  train_features_ac_w5], axis=1)
    classify(sets, train_matrix, "with EXTRA features but without affective contrast" , coef=True, num_feat = train_features.shape[1] )
    
    removed = np.delete(train_matrix, [train_features_dist_avg , train_features_dist_stdev, train_features_dist_min, train_features_dist_max, 
                                  train_features_dist_dif], axis=1)
    classify(sets, train_matrix, "with EXTRA features but without semantic distance" , coef=True, num_feat = train_features.shape[1] )
    
    removed = np.delete(train_matrix, [train_features_dist_avg , train_features_dist_stdev, train_features_dist_min, train_features_dist_max, 
                                  train_features_dist_dif], axis=1)
    classify(sets, train_matrix, "with EXTRA features but without semantic distance" , coef=True, num_feat = train_features.shape[1] )
    
#     log.write("\nMethod = Linear SVM with bag-of-words features\n")
#     acc = []
#     pre = []
#     rec = []
#     f1s = []
#     for set in sets:
#         train_set = set[0]
#         validation_set = set[2]
#         test_set = set[1]
#         
#         best_c = 1.0
#         best_f1 = 0.0
#         for c in c_range:
#             model = LinearSVC( random_state=0, C = c)
#             model.fit( train_matrix[train_set] , train_labels[train_set] )
#             results = model.predict(train_matrix[validation_set])
#             f1 = sklearn.metrics.f1_score( train_labels[validation_set], results )
#             if  f1 > best_f1:
#                 best_c = c
#                 best_f1 = f1
#         log.write('BEST C ' + str(best_c)+'\tBEST F1 ' + str(best_f1)+'\n')       
#                 
#         train_90 = train_set + validation_set
#         train_X_slice = train_matrix[train_90]
#         train_Y_slice = train_labels[train_90]
#         test_X_slice = train_matrix[test_set]
#         test_Y_slice = train_labels[test_set]    
#     
#         model = LinearSVC( random_state=0, C = best_c)
#         model.fit( train_X_slice , train_Y_slice )
#         results = model.predict(test_X_slice)
#         acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
#         pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
#         rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
#         f1s.append(sklearn.metrics.f1_score( test_Y_slice, results )) 
#     log.write("Overall avg accuracy: %.2f\n" % np.mean(acc))
#     log.write("Overall results " + str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
#     
#     log.write('Logistic Regression with bag-of-words features\n')
#     acc = []
#     pre = []
#     rec = []
#     f1s = []
#     for set in sets:
#         train_set = set[0]
#         validation_set = set[2]
#         test_set = set[1]
#         
#         best_c = 1.0
#         best_f1 = 0.0
#         for c in c_range:
#             model = linear_model.LogisticRegression(C = c)
#             model.fit( train_matrix[train_set] , train_labels[train_set] )
#             results = model.predict(train_matrix[validation_set])
#             f1 = sklearn.metrics.f1_score( train_labels[validation_set], results )
#             if f1  > best_f1:
#                 best_c = c
#                 best_f1 = f1
#         log.write('BEST C ' + str(best_c)+'\tBEST F1 ' + str(best_f1)+'\n')       
#                 
#         train_90 = train_set + validation_set
#         train_X_slice = train_matrix[train_90]
#         train_Y_slice = train_labels[train_90]
#         test_X_slice = train_matrix[test_set]
#         test_Y_slice = train_labels[test_set]    
#     
#         model = model = linear_model.LogisticRegression(C = best_c)
#         model.fit( train_X_slice , train_Y_slice )
#         results = model.predict(test_X_slice)
#         acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
#         pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
#         rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
#         f1s.append(sklearn.metrics.f1_score( test_Y_slice, results )) 
#     log.write("Overall avg accuracy: %.2f\n" % np.mean(acc))
#     log.write("Overall results " + str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n\n')
#     
#     ''' 
#     CLASSIFY WITH ADDED EXTRA FEATURES
#     '''
#     train_matrix = np.hstack( (train_matrix,train_features) )
#     num_feat = train_features.shape[1]
#     log.write("EXTRA FEATURES: " +str(num_feat)+'\n\n')
#     
#     log.write("\nMethod = Linear SVM with bag-of-words and EXTRA features\n")
#     acc = []
#     pre = []
#     rec = []
#     f1s = []
#     coef = []
#     for set in sets:
#         train_set = set[0]
#         validation_set = set[2]
#         test_set = set[1]
#         
#         best_c = 1.0
#         best_f1 = 0.0
#         for c in c_range:
#             model = LinearSVC( random_state=0, C = c)
#             model.fit( train_matrix[train_set] , train_labels[train_set] )
#             results = model.predict(train_matrix[validation_set])
#             f1 = sklearn.metrics.f1_score( train_labels[validation_set], results )
#             if f1 > best_f1:
#                 best_c = c
#                 best_f1 = f1   
#         log.write('BEST C ' + str(best_c)+'\tBEST F1 ' + str(best_f1)+'\n')     
#                 
#         train_90 = train_set + validation_set
#         train_X_slice = train_matrix[train_90]
#         train_Y_slice = train_labels[train_90]
#         test_X_slice = train_matrix[test_set]
#         test_Y_slice = train_labels[test_set]    
#     
#         model = LinearSVC( random_state=0, C = best_c)
#         model.fit( train_X_slice , train_Y_slice )
#         results = model.predict(test_X_slice)
#         acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
#         pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
#         rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
#         f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
#         coef.append(model.coef_[0][-num_feat:])
#     log.write("Overall avg accuracy: %.2f\n" % np.mean(acc))
#     log.write("Overall results " + str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n')
#     log.write("Overall Feature Coef:\n"+str(coef)+'\n\n')
#     
#     log.write('Logistic Regression with bag-of-words and EXTRA features\n')
#     coef = []
#     acc = []
#     pre = []
#     rec = []
#     f1s = []
#     for set in sets:
#         train_set = set[0]
#         validation_set = set[2]
#         test_set = set[1]
#         
#         best_c = 1.0
#         best_f1 = 0.0
#         for c in c_range:
#             model = linear_model.LogisticRegression(C = c)
#             model.fit( train_matrix[train_set] , train_labels[train_set] )
#             results = model.predict(train_matrix[validation_set])
#             f1 = sklearn.metrics.f1_score( train_labels[validation_set], results )
#             if  f1 > best_f1:
#                 best_c = c
#                 best_f1 = f1
#         log.write('BEST C ' + str(best_c)+'\tBEST F1 ' + str(best_f1)+'\n')     
#                 
#         train_90 = train_set + validation_set
#         train_X_slice = train_matrix[train_90]
#         train_Y_slice = train_labels[train_90]
#         test_X_slice = train_matrix[test_set]
#         test_Y_slice = train_labels[test_set]    
#     
#         model = linear_model.LogisticRegression(C = best_c)
#         model.fit( train_X_slice , train_Y_slice )
#         results = model.predict(test_X_slice)
#         acc.append(sklearn.metrics.accuracy_score( test_Y_slice , results ))
#         pre.append(sklearn.metrics.precision_score( test_Y_slice, results ))
#         rec.append(sklearn.metrics.recall_score( test_Y_slice, results ))
#         f1s.append(sklearn.metrics.f1_score( test_Y_slice, results ))
#         coef.append(model.coef_[0][-num_feat:]) 
#     log.write("Overall avg accuracy: %.2f\n" % np.mean(acc))
#     log.write("Overall results " + str(np.mean(pre))+'\t'+str(np.mean(rec))+'\t'+str(np.mean(f1s))+'\n')
#     log.write("Overall Feature Coef:\n"+str(coef)+'\n\n')
#     
    

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    log.write('ERROR ' + str(exc_type) + ',' + str(exc_obj) + ',' +fname+',' + str(exc_tb.tb_lineno)+'\n')
    log.close()
log.close()
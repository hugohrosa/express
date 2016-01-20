# -*- coding: utf-8 -*-

import unicodecsv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation

def classify(data,cats,num_pub):
    vect = TfidfVectorizer(analyzer = 'word', stop_words = stopwords)
    tfidf_matrix = vect.fit_transform(data)
    
    print '#### SVM Classifier'
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(tfidf_matrix[:2*num_pub], cats[:2*num_pub])
    predicted = clf.predict(tfidf_matrix[2*num_pub:])
    print (metrics.classification_report(cats[2*num_pub:], predicted))
    conf_matrix = metrics.confusion_matrix(cats[2*num_pub:], predicted)
    print conf_matrix
    print '\n'
    
    print '#### Logistic Regression'
    regr = linear_model.LogisticRegression()
    regr.fit(tfidf_matrix[:2*num_pub], cats[:2*num_pub])
    predicted = regr.predict(tfidf_matrix[2*num_pub:])
    conf_matrix = metrics.confusion_matrix(cats[2*num_pub:], predicted)
    print (metrics.classification_report(cats[2*num_pub:], predicted))
    print conf_matrix 
    print '\n'
    
def biggest(ns,s,n):
    ''' determinar o maior de 3 valores: ns - num anotadores 'não sei', s - num anotadores 'sim', n - num anotadores 'não'
    '''
    Max = ns
    out = u'Não Sei'
    if s > Max:
        Max = s
        out = u'Sim' 
    if n > Max:
        Max = n
        out = u'Não'
        if s > n:
            Max = s
            out = u'Sim'
    return out

def classify_cv(data, cats, k):
    clf = svm.SVC(gamma=0.001, C=100.)
    vect = TfidfVectorizer(analyzer = 'word', stop_words = stopwords)
    tfidf_matrix = vect.fit_transform(data)
    predicted = cross_validation.cross_val_predict(clf, tfidf_matrix, cats, cv=k)
    conf_matrix = metrics.confusion_matrix(cats, predicted)
    print (metrics.classification_report(cats, predicted))

#stopwords = [word.decode('utf-8') for word in stopwords.words('portuguese')]
stopwords = [word for word in stopwords.words('portuguese')]
input = open('data_all.csv', 'rU')
reader = unicodecsv.DictReader(input, encoding = 'utf-8', delimiter = '\t')#, fieldnames = fieldnames )
num_pub = 4185

train = []
train_cats = []
t1_y = []
t1_y_cats = []
t1_n = []
t1_n_cats = []
t2_y = []
t2_y_cats = []
t2_n = []
t2_n_cats = []
ip=0
p=0
all = []
all_c = []
for row in reader:
    if row['fonte']==u'Inimigo Público':
        if ip==num_pub:
            pass
        else:
            if int(row['num_de_anotadores_total'])<=1:
                ip+=1
                train.append(row['texto'])
                train_cats.append(u'Sim')
    elif row['fonte']==u'Público':
        if p==num_pub:
            pass
        else:
            if int(row['num_de_anotadores_total']) == 0:
                train.append(row['texto'])
                train_cats.append(u'Não')
                p+=1
    lbl = biggest(int(row['naosei_ironico']), int(row['sim_ironico']), int(row['nao_ironico']))
    if lbl == u'Sim':
        t2_y.append(row['texto'])
        t2_y_cats.append(lbl)
    elif lbl == u'Não':
        t2_n.append(row['texto'])
        t2_n_cats.append(lbl)

        
print p
print ip

#print '::::::::::::: 1a EXP - Basta Um'        
#data = train+t1_y[:len(t2_n)]+t1_n
#cats = train_cats+t1_y_cats[:len(t1_n_cats)]+t1_n_cats
#classify(data,cats,num_pub)

print '::::::::::::: 2a EXP - Classe mais representada'        
data = train+t2_y[:len(t2_n)]+t2_n
cats = train_cats+t2_y_cats[:len(t2_n_cats)]+t2_n_cats
classify(data,cats,num_pub)

print '::::::::::::: 3a EXP - 10-fold cross validation (todas anotações automáticas, menos os 250)'
classify_cv(train, train_cats, 10)

print '::::::::::::: 4a EXP - 10-fold cross validation (todas anotações manuais, mais s/n das 250)'
classify_cv(data, cats, 10)

print '::::::::::::: 5a EXP - 10-fold cross validation (todas anotações manuais, só s/n das 250)'
data = t2_y[:len(t2_n)]+t2_n
cats = t2_y_cats[:len(t2_n_cats)]+t2_n_cats
classify_cv(data, cats, 10)


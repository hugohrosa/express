# -*- coding: utf-8 -*-

import unicodecsv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics

stopwords = [word for word in stopwords.words('portuguese')]
input = open('data_all.csv', 'rU')
reader = unicodecsv.DictReader(input, encoding = 'utf-8', delimiter = '\t')#, fieldnames = fieldnames )

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

lbl_y = 'Imparidade'
lbl_n = 'Não Imparidade'

split_trn = 1000
split_tst = 133

train = []
train_cats = []
test = []
test_cats = []

imp = 0
n_imp = 0
for row in reader:
    if int(row[lbl_y])>=1 and imp < split_trn:
        #training set for imparity 1000 samples
        imp+=1
        train.append(row['texto'])
        train_cats.append(lbl_y)
    elif int(row[lbl_y])==0 and n_imp < split_trn:
        #training set for not imparity 1000 samples
        n_imp+=1
        train.append(row['texto'])
        train_cats.append(lbl_n)
    elif int(row[lbl_y])>=1 and imp >= split_trn  and imp < split_trn + split_tst:
        #test set for imparity 133 samples
        imp+=1
        test.append(row['texto'])
        test_cats.append(lbl_y)
    elif int(row[lbl_y])==0 and n_imp >= split_trn  and n_imp < split_trn + split_tst:
        #test set for not imparity 133 samples
        n_imp+=1
        test.append(row['texto'])
        test_cats.append(lbl_n)
        
data = train+test
cats = train_cats+test_cats
classify(data,cats,split_trn)


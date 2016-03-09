# -*- coding: utf-8 -*-
'''
Created on Jan 18, 2016

@author: hugohrosa
'''

import urllib,os, sys, re
import nltk
import json
import unicodecsv
import cPickle
#from ipdb import set_trace
import time
import itertools
from bs4 import BeautifulSoup
import requests
from selenium import webdriver 
import re
import codecs
import string
# import requests
# 
# from PyQt4.QtGui import *  
# from PyQt4.QtCore import *  
# from PyQt4.QtWebKit import *  
# from lxml import html 

log = codecs.open('_dispair_log_coocurs_sapo.txt','w','utf-8')

# class Render(QWebPage):  
#   def __init__(self, url):  
#     self.app = QApplication(sys.argv)  
#     QWebPage.__init__(self)  
#     self.loadFinished.connect(self._loadFinished)  
#     self.mainFrame().load(QUrl(url))  
#     self.app.exec_()  
#   
#   def _loadFinished(self, result):  
#     self.frame = self.mainFrame()  
#     self.app.quit()

# def getSearchNumHits(url):
#     driver = webdriver.PhantomJS()   
#     try:
#         driver.get(url) 
#         page_html = driver.execute_script('return document.documentElement.innerHTML;')
#  
#     except Exception:
#         log.write('Webdriver Error ' + url+'\n')
#         #driver.quit()
#         return 0 
#     #r = Render(url)  
#     #result = r.frame.toHtml()
#     #page_html = str(result.toAscii())
#     soup = BeautifulSoup(page_html, 'html.parser')
#     search_results = soup.find('div', {'id': 'resInfo-0'})
#     if search_results is not None:
#         hits = numbersOnly(search_results.text.partition('(')[0])
#         log.write('\t OK \t ' + url + ' ' + str(hits)+'\n')
#     else:
#         log.write('\t FAILED \t ' + url+'\n')
#         hits = 0
#     driver.quit()
#     return hits

def getSearchNumHitsVoxx(url):
    try:
        log.write('\tgetSearchNumHitsVoxx Vou abrir link')       
        #response = urllib.urlopen(url)
        response = requests.get(url)
        #log.write('\tVou ler conteúdo')
        #content = response.read()
        if response.status_code != 200:
            log.write('\tFAILED ' + str(response.status_code) + '\n')
            return None
        log.write('\tVou carregar JSON')
        #data = json.loads(content)
        data = json.loads(response.content)
        if len(data.values()[0])==0:
            log.write('\tFAILED ' + url + '\n')
            return None
        else:
            hits = data['response']['numFound']
            log.write('\tOK ' + str(hits) + ' ' + url + '\n')
            return hits
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        log.write('\tDecoding JSON has failed\t'+ str(url) +'\n')
        pass
    except:
        log.write('\tGeneral error getSearchNumHitsVoxx: ' + str(url) + ' ' + str(sys.exc_info()[0])+ ' ' + str(sys.exc_info()[1]) + ' ' + str(sys.exc_info()[2])+'\n')
        pass

def numbersOnly(text):
    return re.sub('[^0-9]','',text)

def PoI(entity, poi, err):
    try:       
        #sys.stdout.write("\tlookup: %s\n" % entity)
        #sys.stdout.flush()
        #log.write("\tlookup: "+ entity +"\n")
        url = 'http://services.sapo.pt/InformationRetrieval/Verbetes/WhoIs?name='+entity.encode('utf-8')
        log.write('\tPoI Vou abrir link')
        #response = urllib.urlopen(url)
        response = requests.get(url)
        #log.write('\tVou ler conteúdo')
        #content = response.read()
        if response.status_code != 200:
            log.write('\tFAILED ' + str(response.status_code) + '\n')
            return None
        log.write('\tVou carregar JSON')
        #data = json.loads(content)
        data = json.loads(response.content)
        if len(data.values()[0])==0:
            return ''
        else:
            for x in data['verbetes']:
                if x['officialName'] in poi:
                    return ''
            return data['verbetes'][0]['officialName']
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        log.write('\tDecoding JSON has failed\t'+str(url)+'\n')
        pass
    except:
        log.write('\tGeneral error PoI: ' + str(url) + ' ' + str(sys.exc_info()[0])+ ' ' + str(sys.exc_info()[1]) + ' ' + str(sys.exc_info()[2])+'\n')
        pass


data = open('DATA/data_all.csv', 'rU')
reader = unicodecsv.DictReader(data, encoding = 'utf-8', delimiter = '\t')#, 
with open("Models/tagger.pkl") as fid:
    tagger_fast = cPickle.load(fid)
    
# with open("DATA/entities_by_sentence_sapo.pkl") as fid:
#     ent_dict= cPickle.load(fid)
# with open("DATA/coocurs_sapo.pkl") as fid:
#     co_occurs = cPickle.load(fid)

noun_list = [u'NOUN',u'N']
err = 0
co_occurs = {}
sent_id = 0
prev_requests = {}
t0 = time.time()
skipURL=[]
exclude = set(unicode(string.punctuation))
exclude.add(u'“')
ent_dict = dict()
for row in reader:
    if int(row['num_de_anotadores_total'])>=1: 
                  
#         if sent_id < 2540:
#             print "\tbailed! " + str(sent_id)
#             sent_id+=1
#             continue
        sent_id+=1
        entities = []
        
        #remove punctuation and numbers from text string
        #text = re.sub(ur"\p{P}+", '', row['texto'])
        text = ''.join(ch for ch in row['texto'] if ch not in exclude)
        log.write(str(sent_id)+' - ' + text+'\n')
        
        #POS tagging
        sent_tagged = tagger_fast.tag(nltk.word_tokenize(text))
        sent_tagged = [w for w in sent_tagged if w[0] not in nltk.corpus.stopwords.words('portuguese')] #unicode(nltk.corpus.stopwords.words('portuguese'))]
        # print "SENTENCE:",  sent_tagged            
        poi = []
        skip_next=False
        for w in sent_tagged:
            token = w[0]
            pos = w[1]
            #if the token only has one character, it doesn't interest us
            if len(token)<2:
                continue
            #if the last entity was a bi-gram skip this word
            if skip_next:
                #log.write("\tskipped:" + token + "\n")
                skip_next=False
                continue
            #ignore the last token!? Why?
            
            if pos in noun_list:
                if sent_tagged.index(w)+1 < len(sent_tagged):
                #if the next token is also a NOUN combine the tokens
                    n_pos = sent_tagged[sent_tagged.index(w)+1][1]
                    if n_pos in noun_list:
                        token_cmp = token + ' ' + sent_tagged[sent_tagged.index(w)+1][0]
                        entity=None
                        while entity is None:
                            #entity = PoI(token, poi, err)
                            entity = PoI(token_cmp, poi, err)
                            if entity is None:
                                print str(token_cmp) + ' is None'
                        if entity != '' and token not in co_occurs:
                            #print entity
                            skip_next=True
                            token = entity
                #print '\tadded: ' + token
                if token.lower() not in entities:
                    entities.append(token.lower())
                    
            #avoid sending the same requests    
            if token in prev_requests or token in co_occurs or pos not in [u'NOUN',u'N']: 
                #print "\tignored: %s" % token
                continue
        log.write('\n\t' + str(entities) + '\n')
        ent_dict[sent_id]={'entities':entities, 'imparity':row['Imparidade']}
        
        #make combinations of entities for Sapo search engine
        entity_pairs = itertools.combinations(entities, 2)      
        for e_1, e_2 in entity_pairs:
            e_a = e_1.encode('utf-8')
            e_b = e_2.encode('utf-8')
            total_occurences_a=0
            total_occurences_b=0
            #url_sapo_co_ab = 'http://www.sapo.pt/pesquisa?q="'+urllib.quote(e_a)+'" + "'+urllib.quote(e_b)+'"'##gsc.tab=0&gsc.q="'+e_a+'" + "'+e_b+'"&gsc.page=1'
            url_sapo_co_ab = 'http://services.sapo.pt/InformationRetrieval/News/Search?q="'+urllib.quote(e_a)+'" "'+urllib.quote(e_b)+'"&ESBUsername=popstar@users.sdb.sapo.pt&ESBPassword=DsEsfkesd6n2fwWds02&wt=json'
            #co_hits_ab = getSearchNumHits(url_sapo_co_ab)
            co_hits_ab=None
            while co_hits_ab is None:
                co_hits_ab = getSearchNumHitsVoxx(url_sapo_co_ab)
                if co_hits_ab is None:
                    print str(url_sapo_co_ab) + ' is None'
            if e_a not in skipURL:
                #url_sapo_a = 'http://www.sapo.pt/pesquisa?q="'+urllib.quote(e_a)+'"'##gsc.tab=0&gsc.q="'+e_a+'"&gsc.page=1'
                url_sapo_a = 'http://services.sapo.pt/InformationRetrieval/News/Search?q="'+urllib.quote(e_a)+'"&ESBUsername=popstar@users.sdb.sapo.pt&ESBPassword=DsEsfkesd6n2fwWds02&wt=json'
                total_occurences_a = None
                while total_occurences_a is None:
                    total_occurences_a = getSearchNumHitsVoxx(url_sapo_a)
                    if total_occurences_a is None:
                        print str(url_sapo_a) + ' is None'
            if e_b not in skipURL:
                #url_sapo_b = 'http://www.sapo.pt/pesquisa?q="'+urllib.quote(e_b)+'"'##gsc.tab=0&gsc.q="'+e_b+'"&gsc.page=1'
                url_sapo_b = 'http://services.sapo.pt/InformationRetrieval/News/Search?q="'+urllib.quote(e_b)+'"&ESBUsername=popstar@users.sdb.sapo.pt&ESBPassword=DsEsfkesd6n2fwWds02&wt=json'
                total_occurences_b = None
                while total_occurences_b is None:
                    total_occurences_b = getSearchNumHitsVoxx(url_sapo_b)
                    if total_occurences_b is None:
                        print str(url_sapo_b) + ' is None'
            
            if co_occurs.has_key(e_a):
                co_item = co_occurs[e_a]['verbetes']
                if not co_item.has_key(e_b):
                    co_item[e_b]=co_hits_ab
                co_occurs[e_a]['verbetes']=co_item   
            else:
                co_item_ab = dict()
                co_item_ab[e_b]=co_hits_ab
                co_occurs[e_a] = {"official_name": e_a, "verbetes": co_item_ab, "total_occurences": total_occurences_a}
            if co_occurs.has_key(e_b):
                co_item = co_occurs[e_b]['verbetes']
                if not co_item.has_key(e_a):
                    #co_item[e_a]=co_hits_ba
                    co_item[e_a]=co_hits_ab
                co_occurs[e_b]['verbetes']=co_item
            else:
                co_item_ba = dict()               
                co_item_ba[e_a]=co_hits_ab
                co_occurs[e_b] = {"official_name": e_b, "verbetes": co_item_ba, "total_occurences": total_occurences_b}
            skipURL.append(e_a)
            skipURL.append(e_b)
        #print co_occurs
        with open("DATA/_dispair_sentences_sapo.pkl","wb") as fid:
            cPickle.dump(ent_dict, fid)
        with open("DATA/_dispair_cooccurs_sapo.pkl","wb") as fid:
            cPickle.dump(co_occurs, fid)
        print "Sentence "+ str(sent_id) + " done at " + time.strftime('%X %x %Z') + " after %d minutes running" % ((time.time()-t0 )/60)

#print 'terminei queries'

print "DONE! :) "
print "Took %d minutes" % ((time.time()-t0 )/60)
 
sys.exit()



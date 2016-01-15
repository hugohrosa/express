# -*- coding: utf-8 -*-

import csv
import unicodecsv
import codecs
import os, errno

ip = codecs.open('_semtag_dataset_webanno_tfidf_inimigo.txt','r','utf-8')
pub = codecs.open('_semtag_dataset_webanno_tfidf_publico.txt','r','utf-8')

ip_array = []
ip_full = []
for l in ip.readlines():
    ip_full.append(l.replace('\n',''))
    l = ''.join(e for e in l if e.isalnum())
    ip_array.append(l)

pub_array = []
pub_full = []
for l in pub.readlines():
    pub_full.append(l.replace('\n',''))
    l = ''.join(e for e in l if e.isalnum())
    pub_array.append(l)


def checkOrigin(text):
    #text_s = "".join(text.split())
    text_s = ''.join(e for e in text if e.isalnum())
    if text_s in ip_array:
        return 'Inimigo Público'
    elif text_s in pub_array:
        return 'Público'
    print 'SEM FONTE:'+ text
    return 'no source'

annotators = ['hanna','cristina','cfreitas','ccarvalho','andrea']
fieldnames = ['texto', 'fonte', 'ironico', 'num_de_anotadores_ironico', 'num_de_anotadores_total', 'Comparação', 'Hipérbole', 'Imparidade', 'Metáfora', 'Paradoxo', 'Vulgarismo', 'Outro', 'Sem Evidência']
#output = codecs.open('annotated_10_INST.txt','w','utf-8')
#output = codecs.open('annotation_stats/data.txt','wb','utf-8')
output = open('data_all.csv','wb')
csvw = unicodecsv.DictWriter(output, delimiter = '\t',fieldnames = fieldnames)
final = dict()

filename = 'block_0_IRONIA'
for an in annotators:
    try:
        with open('express_precuration/annotation/'+filename+'.tcf/'+an+'.tsv') as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                if len(line) > 0:
                    if line[0].startswith('#text='):
                        text = line[0].rsplit('[ [',1)[0]
                        text = text.replace('#text=','').decode('utf-8')
                        if not final.has_key(text):
                            origin = checkOrigin(text)
                            final[text] = {'texto':text, 'fonte':origin, 'ironico':'--', 'num_de_anotadores_ironico': 0, 'num_de_anotadores_total':0,
                                           'Comparação':0, 'Hipérbole':0, 'Imparidade':0, 'Metáfora':0, 'Paradoxo':0, 'Vulgarismo':0, 'Outro':0, 'Sem Evidência':0} 
                    if len(line)==4 and line[2] != '_':
                        aux_d = final[text]
                        aux_d['num_de_anotadores_total']+=1
                        if line[2]=='Sim': 
                            aux_d['num_de_anotadores_ironico']+=1
                        final[text]=aux_d
    except IOError, ioex:
        if ioex.errno==2:
            continue
        else:
            break

filename = 'block_0_INSTRUMENTO'
for an in annotators:
    try:
        with open('express_precuration/annotation/'+filename+'.tcf/'+an+'.tsv') as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                if len(line) > 0:
                    if line[0].startswith('#text='):
                        text = line[0].rsplit('[ [',1)[0]
                        text = text.replace('#text=','').decode('utf-8')
                        if not final.has_key(text):
                            origin = checkOrigin(text)
                            final[text] = {'texto':text, 'fonte':origin, 'ironico':'--', 'num_de_anotadores_ironico': 0, 'num_de_anotadores_total':0,
                                           'Comparação':0, 'Hipérbole':0, 'Imparidade':0, 'Metáfora':0, 'Paradoxo':0, 'Vulgarismo':0, 'Outro':0, 'Sem Evidência':0} 
                    if len(line)==4 and line[2] != '_':
                        cat = line[2].split('|')
                        aux_d = final[text]
                        for c in cat:
                            if aux_d.has_key(c):
                                aux_d[c]+=1
                        final[text]=aux_d
    except IOError, ioex:
        if ioex.errno==2:
            continue
        else:
            break                      

for i in range(0,10,1):
    filename = 'block_100'+str(i)+'_INSTRUMENTO'
    for an in annotators:
        try:
            with open('express_precuration/annotation/'+filename+'.tcf/'+an+'.tsv') as tsv:
                for line in csv.reader(tsv, dialect="excel-tab"):
                    if len(line) > 0:
                        if line[0].startswith('#text='):
                            text = line[0].rsplit('[ [',1)[0]
                            text = text.replace('#text=','').decode('utf-8')
                            if not final.has_key(text):
                                origin = checkOrigin(text)
                                final[text] = {'texto':text, 'fonte':origin, 'ironico':'--', 'num_de_anotadores_ironico': 0, 'num_de_anotadores_total':0,
                                               'Comparação':0, 'Hipérbole':0, 'Imparidade':0, 'Metáfora':0, 'Paradoxo':0, 'Vulgarismo':0, 'Outro':0, 'Sem Evidência':0} 
                        if len(line)==4 and line[2] != '_':
                            cat = line[2].split('|')
                            aux_d = final[text]
                            aux_d['num_de_anotadores_total']+=1
                            for c in cat:
                                if aux_d.has_key(c):
                                    aux_d[c]+=1
                            final[text]=aux_d
        except IOError, ioex:
            if ioex.errno==2:
                continue
            else:
                break

keys_list = []
for k in final.keys():
    keys_list.append(''.join(e for e in k if e.isalnum()))
    
for p in pub_array:
    if p not in keys_list:
        text = pub_full[pub_array.index(p)]
        final[text] = {'texto': text, 'fonte':'Público', 'ironico':'--', 'num_de_anotadores_ironico': 0, 'num_de_anotadores_total':0,
                        'Comparação':0, 'Hipérbole':0, 'Imparidade':0, 'Metáfora':0, 'Paradoxo':0, 'Vulgarismo':0, 'Outro':0, 'Sem Evidência':0}
        
for ip in ip_array:
    if ip not in keys_list:
        text = ip_full[ip_array.index(ip)]
        final[text] = {'texto':text, 'fonte':'Inimigo Público', 'ironico':'--', 'num_de_anotadores_ironico': 0, 'num_de_anotadores_total':0,
                        'Comparação':0, 'Hipérbole':0, 'Imparidade':0, 'Metáfora':0, 'Paradoxo':0, 'Vulgarismo':0, 'Outro':0, 'Sem Evidência':0}

csvw.writeheader()
for k,v in final.iteritems():
    if final[k]['num_de_anotadores_total']>1 and final[k]['num_de_anotadores_ironico'] / float(final[k]['num_de_anotadores_total']) >= 0.5:
        final[k]['ironico']='Sim'
    elif final[k]['num_de_anotadores_total']>1 and final[k]['num_de_anotadores_ironico'] / float(final[k]['num_de_anotadores_total']) < 0.5:
        final[k]['ironico']='Não'
    #else:
    #    final[k]['ironico']='--'
    csvw.writerow(final[k])
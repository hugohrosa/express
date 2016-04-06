'''
Created on Mar 31, 2016

@author: hugohrosa
'''

import csv
from googleapiclient.discovery import build

fieldnames = ['', 'Word','V.Mean.Sum','V.SD.Sum','V.Rat.Sum','A.Mean.Sum','A.SD.Sum','A.Rat.Sum','D.Mean.Sum','D.SD.Sum','D.Rat.Sum',
              'V.Mean.M','V.SD.M','V.Rat.M','V.Mean.F','V.SD.F','V.Rat.F','A.Mean.M','A.SD.M','A.Rat.M','A.Mean.F','A.SD.F','A.Rat.F',
              'D.Mean.M','D.SD.M','D.Rat.M','D.Mean.F','D.SD.F','D.Rat.F','V.Mean.Y','V.SD.Y','V.Rat.Y','V.Mean.O','V.SD.O','V.Rat.O',
              'A.Mean.Y','A.SD.Y','A.Rat.Y','A.Mean.O','A.SD.O','A.Rat.O','D.Mean.Y','D.SD.Y','D.Rat.Y','D.Mean.O','D.SD.O','D.Rat.O',
              'V.Mean.L','V.SD.L','V.Rat.L','V.Mean.H','V.SD.H','V.Rat.H','A.Mean.L','A.SD.L','A.Rat.L','A.Mean.H','A.SD.H','A.Rat.H',
              'D.Mean.L','D.SD.L','D.Rat.L','D.Mean.H','D.SD.H','D.Rat.H']
writer = csv.DictWriter(open("Irony Text Classification/Ratings_Warriner_et_al_translated.csv","w"), fieldnames = fieldnames)
writer.writeheader()

service = build('translate', 'v2','')
            #developerKey='AIzaSyAIXwQSW0YbUUuMGH6YfPQw1116FqtGVFg')

i=0
translated = []
for row in csv.DictReader(open("Irony Text Classification/Ratings_Warriner_et_al.csv")): 
    en_word = row["Word"].lower()
    result = service.translations().list(source='en',target='pt', q=en_word).execute()
    pt_word = result['translations'][0]['translatedText'].lower()

    duplicate = False
    if pt_word != en_word:
        for d in translated:
            if pt_word in d.values():
                duplicate = True
                break
            else:
                duplicate = False
        if not duplicate:
            row["Word"]=pt_word
            translated.append(row)

for d in translated:
    writer.writerow(d)

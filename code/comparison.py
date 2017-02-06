'''
Created on Mar 31, 2016

@author: hugohrosa
'''

'''
COMPARATIVAS (estabelecem comparação): 
que, do que (depois de mais, maior, melhor ou menos, menor, pior), como...
Também as locuções: tão...como, tanto...como, mais...do que, menos...do que, assim como, bem como, que nem...
'''

import re

como1 = False
como2 = False

sequential = ['assim como','bem como','que nem']

que_first = ['mais', 'maior', 'melhor', 'menos', 'menor','pior']
que_last = ['do que', 'que']

como_first = ['tão', 'tanto']
como_last = ['como']

text = 'és mais do que um anormal'

for q1 in que_first:
    for match in re.finditer(q1, text):
        print(q1 + ' ' + str(match.start()))
    for q2 in que_last:
        for match in re.finditer(q2, text):
            print(q2 + ' ' + str(match.start()))

# for c1 in como_first:
#     if c1 in text:
#         como1 = True
# for c2 in como_last:
#         if c2 in text:
#             como2 = True
# 
# for q1 in que_first:
#     if c1 in text:
#         que1 = True
# for q2 in que_last:
#     if c2 in text:
#         que2 = True
#         
# for s in sequential:
#     if s in text:
#         seq = True
#         
# if (como1 and como2) or (que1 and que2) or seq:
#     print('Comparação')        
    
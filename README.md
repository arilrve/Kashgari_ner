﻿# Kashgari_ner
 
with open('data/sample.data','r', encoding='utf8') as f:
    data = f.read()
    
all_data = []
tmp = []
for i in data :
    if i.find('。') :
        tmp.append(i)
        all_data.append(tmp)
        tmp = []
        continue
    else :
        tmp.append(i)
        
print(all_data)  

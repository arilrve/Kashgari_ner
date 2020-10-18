import kashgari
from kashgari import utils
import re
import pickle

loaded_model = kashgari.utils.load_model('Model')
def cut_text(text, lenth):
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    return textArr


def extract_labels(text, ners):
    ner_reg_list = []
    if ners:
        new_ners = []
        for ner in ners:
            new_ners += ner;
        for word, tag in zip([char for char in text], new_ners):
            if tag != 'O':
                ner_reg_list.append((word, tag))

    # 輸出模型的NER識別结果
    labels = {}
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                label = ""
                end = i + 1
                while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                    end += 1
                ner_type = item[1].split('-')[1]
                if ner_type not in labels.keys():
                    labels[ner_type] = []
                label += ''.join([item[0] for item in ner_reg_list[i:end]])
                labels[ner_type].append(label)
    return labels
# open development
with open("development_1.txt","r",encoding="UTF-8") as f:
    datas = f.read()
# 分割 "--------------------"
datas = datas.split("--------------------")

# 文章分割
articles = []
for i in range(len(datas)):
    datas[i] = datas[i].strip("\n\n")
    datas[i] = datas[i].split("\n")
    articles.append(datas[i][1])

    
output = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"

# 依據predict結果統整資料
def get_data(all_tokens,article,article_id):
    global output
    word_num = 0
    start_position = 0
    end_position = 0
        
    for i in range(len(all_tokens)) :
        # print(all_tokens[i][0])
        try:
            if all_tokens[i][0] == "B" :
                start_position = word_num
                # print(start_position)
                entity_type = all_tokens[i][2:]
            elif start_position is not None and all_tokens[i][0] =="I" and all_tokens[i+1][0]=='O' :
            
                end_position = word_num
                entity_text = article[start_position:end_position+1]
                
                line = str(article_id)+'\t'+str(start_position)+'\t'+str(end_position+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
        except :
            end_position = word_num
            entity_text = article[start_position-1:end_position]
            # print(article_id,entity_text)
            line = str(article_id)+'\t'+str(start_position)+'\t'+str(end_position+1)+'\t'+entity_text+'\t'+entity_type
            output+=line+'\n'

        word_num += 1   
        
    return output

article_id = 0

for article in articles:
    all_token = []
    text_input = article
    texts = cut_text(text_input,100)
    ners = loaded_model.predict([[char for char in text] for text in texts])

    for ner in ners:
        all_token.extend(ner)
    
    output = get_data(all_token,article,article_id)
    article_id+=1
# print(output)

output_path='output.tsv'
with open(output_path,'w',encoding='utf-8') as f:
    f.write(output)
            

# print("NERS : ",ners)
# labels = extract_labels(text_input, ners)
# print("Labels : ",labels)

# while True:
#     text_input = input('sentence: ')
#     texts = cut_text(text_input, 100)
#     ners = loaded_model.predict([[char for char in text] for text in texts])
#     print(ners)
#     labels = extract_labels(text_input, ners)
#     print(labels)
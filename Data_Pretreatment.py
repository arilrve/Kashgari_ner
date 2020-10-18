import pickle

with open("train_sample.data","rb") as f:
    data = f.read().decode("utf-8")
# print(data)
train_data = data.replace("\n","")
train_data = train_data.split("\r")

# 將每段文章分割 
article = []
article_tmp = []

for data in train_data :
    if len(data) == 0 :
        article.append(article_tmp)
        article_tmp = []
        continue
    else :
        article_tmp.append(data)

# 將文章每次遇到句號分割成單獨句子
article_sentence = []
article_tmp = []

for i in range(len(article)) :
    for article_word in article[i] :
        if '。' in article_word :
            article_tmp.append(article_word)
            article_sentence.append(article_tmp)
            article_tmp = []
        else :
            article_tmp.append(article_word)

# 每個sentence的"字元"和"符號"分開加入x、y _data

x_data , y_data = [] , []
x_tmp ,y_tmp = [],[]

for i in range(len(article_sentence)):
    for word in article_sentence[i]:
        x_tmp.append(word[0]) # tmp是一句sentence

    x_data.append(x_tmp) # sentence 加入至data
    x_tmp = []


for i in range(len(article_sentence)):
    for word in article_sentence[i]:
        y_tmp.append(word[2:]) # tmp是一句sentence

    y_data.append(y_tmp) # sentence 加入至data
    y_tmp = []   

def split_data(x_data,y_data,split_rate) :
    x_train = []
    y_train = []

    x_validation = []
    y_validation = []

    x_split_start = len(x_data) - len(x_data) * split_rate
    x_split_start = int(x_split_start)
    
    y_split_start = len(y_data) - len(y_data) * split_rate
    y_split_start = int(y_split_start)

    for i in range(len(x_data)):
        if i < x_split_start :
            x_train.append(x_data[i])
        else :
            x_validation.append(x_data[i])
            

    for i in range(len(y_data)):
        if i < y_split_start :
            y_train.append(y_data[i])
        else :
            y_validation.append(y_data[i])  

    return x_train ,x_validation ,y_train ,y_validation

x_train , x_validation ,y_train ,y_validation = split_data(x_data,y_data,0.2)

pre_data = ( x_train,
             x_validation,
             y_train,
             y_validation
        )

with open('data.pickle','wb') as f :
    pickle.dump(pre_data, f)
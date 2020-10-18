import pickle
import kashgari
from kashgari.embeddings import BertEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
import tensorflow as tf

with open('data.pickle', 'rb') as f:
    data_dic = pickle.load(f)

x_train = data_dic[0]
x_validation = data_dic[1]
y_train = data_dic[2]
y_validation = data_dic[3]

embedding = BertEmbedding('bert-base-chinese',
                            sequence_length = 128)
model = BiLSTM_CRF_Model(embedding)

model.fit(  x_train = x_train,
            x_validate = x_validation,
            y_train = y_train,
            y_validate = y_validation,
            epochs=5,
            batch_size=32,
            )
model.save('Model')
model.evaluate(x_data=x_validation,y_data=y_validation)
from gensim.models.word2vec import Word2Vec

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense


train_df = pd.read_pickle('train.pkl').sample(frac = 1, random_state = 123)
test_df = pd.read_pickle('test.pkl')
w2v_model = Word2Vec.load('word2vec.model')

#將詞向量模型構建為embedding layer

embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items())+1,w2v_model.vector_size))
word2idx = {}
vocab_list = [(word,w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
	word, vec = vocab
	embedding_matrix[i+1] =vec
	word2idx[ word ] = i+1
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)
def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)


PADDING_LENGTH = 200
X = text_to_index(train_df.text)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
print("Shape:", X.shape)
print("Sample:", X[0])

# 在多标签分类问题中， Y值必须被处理成one-hot vector
Y = to_categorical(train_df.category)
print("Shape:", Y.shape)
print("Sample:", Y[0])

#模型的建构，训练，预测
def new_model():
	model = Sequential()
	model.add(embedding_layer)
	model.add(GRU(16))
	model.add(Dense(100,activation = 'relu'))
	model.add(Dense(100,activation = 'relu'))	
	model.add(Dense(100,activation = 'relu'))
	# model.add(Dense(100,activation = 'relu'))
	model.add(Dense(100,activation = 'relu'))

	model.add(Dense(10,activation = 'softmax'))

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	return model

# 用summary()检验建构好的模型
model = new_model()
model.summary()
# 调整validation_split决定要取多少比例的训练集作Holdout验证？？？？？？？？？？？
model.fit(x= X, y= Y, batch_size = 3000, epochs = 100, validation_split = 0.1)



# 待模型训练好，我们需要先对测试集的资料作处理
X_test = text_to_index(test_df.text)
X_test = pad_sequences(X_test, maxlen = PADDING_LENGTH)
# 模型的預測值是10維的向量，每一維分別代表各個category的機率

Y_preds = model.predict(X_test)
print("Shape:" , Y.shape)
print("Sample:" , Y_preds[0])

Y_preds_label = np.argmax(Y_preds, axis= 1)
print("Shape:", Y_preds_label.shape)
print("Sample:" , Y_preds_label[0])

submit = test_df[['id']]
submit['category'] = Y_preds_label
submit.to_csv("submit_csv", index = False)

# #Declare model
# model = word2vec.word2vec(sentences, size = 250)
# #• sentences:The sentences iterable can be simply a list of lists of tokens
# #• size:Dimensionality of the feature vectors
# #• alpha:The initial learning rate
# #• sg:Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used
# #• window:The maximum distance between the current and predicted word within a sentence
# #• workers:Use these many worker threads to train the model
# #• min_count:Ignores all words with total frequency lower than this
# model.save("word2vec.model")

# model.most_similar()

# model.most_similarity(x,y)


# #import
# #declare-use RNN
# #declare Model

# model = Sequential()

# model.add(embedding_layer)
# model.add(SimpleRNN( output_dim = 50,  unroll = True))
# model.add(Dense(OUTPUT_SIZE))
# model.add(Activation('softmax'))


# #Embedding_layer
# #Load the word_bedding

# #model compile
# model.compile(optimizer = adam, loss = 'categorical_crossentropy',metrics = ['accuracy'])

# #train
# #teating

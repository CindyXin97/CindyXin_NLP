import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim. models import word2vec
import logging

train_df = pd.read_pickle('train.pkl')
test_df = pd.read_pickle('test.pkl')

# print(train_df.head())
# print(train_df.sample(10))
print(train_df.columns)
train_df.columns = train_df.columns.str.strip()
print(train_df.text)
print(test_df)




# 串聯 training 與 testing 中的 text，再作洗牌（避免 category 順序影響到結果）
corpus = pd.concat([train_df.text, test_df.text])
corpus = corpus.sample(frac=1)
print(corpus.head())

#把处理好的2D资料结构，直接扔进Word2Vec就OK
model = Word2Vec(corpus, size = 250, iter = 10,  window = 8, min_count = 1, sg = 1) #首先定义model
# model = Word2Vec(corpus, size = 220, iter = 10,  window = 9, min_count = 1, sg = 1) #首先定义model
# model = Word2Vec(corpus, size = 250, iter = 10,  window = 9, min_count = 1, sg = 1) #最後一次

# 检验训练好的词向量模型
def most_similar(w2v_model, words, topn = 10):
	similar_df = pd.DataFrame()
	for word in words:
		# try:
		# 	similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn = topn), columns = [word, 'cos'])
		# 	similar_df = pd.concat([similar_df, similar_words], axis = 1)
		try:
			similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
			similar_df = pd.concat([similar_df, similar_words], axis=1)#将所得到的相似词汇串联在一起
			print(similar_df)
		except:
			print(word, "not found in Word2Vec model!")
	return similar_df

#实例

most_similar(model, [' 懷孕 ',' 網拍 ', ' 補習 ', ' 東京 ', ' XDD ', ' 金宇彬 ',' 化妝品 ', ' 奧斯卡 ', ' 主管 ', ' 女孩 '])
model.save('word2vec.model')
model = Word2Vec.load('word2vec.model')


# def main():

#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     sentences = word2vec.LineSentence(train_df)
#     model = word2vec.Word2Vec(sentences, size=250)

#     #保存模型，供日後使用
#     model.save("word2vec.model")

#     #模型讀取方式
#     # model = word2vec.Word2Vec.load("your_model_name")

# if __name__ == "__main__":
# 	main()
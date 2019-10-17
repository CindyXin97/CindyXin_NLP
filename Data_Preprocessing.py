import os
import pandas as pd

TRAINING_PATH = 'HW3/training/'
TESTING_PATH = 'HW3/testing/'

#取得分类名称与编号

#categories = [dirname for dirname in os.listdir(TRAINING_PATH) if dirname[-4, ] != '_cut'] #涉及路径
#print(len(categories), str(categories))
# categories = [dirname for dirname in os.listdir(TRAINING_PATH) if dirname[-4:] != '_cut']
# print(len(categories), str(categories))

#利用table将用于将文章分类的名称转变为编号
#category2idx = {'Japan_Travel': 3, 'KR_ENTERTAIN': 5,'babymother': 0, 'e-shopping': 1, 'graduate': 2, 'joke': 4}
# category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job':  3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}
#处理每个分类的txt档
# train_list=[]

# default = 'Scruffy'

# for category in categories:
# 	#category_idx = category2dix[category]
# 	category_idx = category2idx[category]
# 	#category_idx = category2idx.get('category', default)????????????

# 	category_path = TRAINING_PATH + category +'_cut/'#路径

# 	for filename in os.listdir(category_path):#路径
# 		file_path = category_path + filename#路径

# 		with open(file_path, encoding = 'utf-8') as file:
# 			words = file.read().strip().split('/')#读取档案，strip()目的在于消除空格以及换行，split()进行切分词组
# 			train_list.append([words, category_idx])#这跟R语言很像，中括号里面进行拓展

# #将train_list转为dataframe
# train_df = pd.DataFrame(train_list, columns = ["text", "category"])
# print("trian_df's Shape:" ,train_df.shape)
# print(train_df.sample(5))


categories = [dirname for dirname in os.listdir(TRAINING_PATH) if dirname[-4:] != '_cut']
print(len(categories), str(categories))
category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job':  3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}
train_list = []

for category in categories:
	category_idx = category2idx[category]
	category_path = TRAINING_PATH + category + '_cut/'

	for filename in os.listdir(category_path):
		filepath = category_path + filename

		with open(filepath, encoding='utf-8') as file:
			words = file.read().strip().split('/')
			train_list.append([words, category_idx])

train_df = pd.DataFrame(train_list, columns=["text", "category"])
print("Shape:", train_df.shape)
train_df.sample(5)
# 关于.DS_Store文件的问题
# 因为之前我的data只有现在data的几个门类
# 但是我用的是参考档案中的语句
# 于是出现了./DS_Store的问题
# 之后我把语句改为如下例子
# category2idx = {'Japan_Travel': 3, 'KR_ENTERTAIN': 5,'babymother': 0, 'e-shopping': 1, 'graduate': 2, 'joke': 4}
# 结果还是会报错关于./DS_Store


# !!!!!!!!!!!!!因为我们最终要对文件进行操作，我们选择文件格式为pickle文件
# 以Pickle格式存储Dataframe
# Pickle可以直接儲存任何的Object 
# 如果用CSV，Dataframe裡的list會變成string
train_df.to_pickle('train.pkl')
train_pickle_df = pd.read_pickle('train.pkl')#??????????????为什么非要读出来？？？？
# print(train_df.equals('train.pkl'))
print("train_pickle_df's Shape", train_pickle_df.shape)

print(train_df.equals('pickle_df'))#？？？？？？为什么一直是false？？？？如何检验？？？
print(train_pickle_df.equals('train.pkl'))

print(train_df.describe())
print(train_pickle_df.describe())

train_df.to_pickle('train.pkl')
pickle_df = pd.read_pickle('train.pkl')
print(train_df.equals(pickle_df))
# 首先回顾一下之前的做法：通过通过TRAIN_PATH获取了categories，然后对其进行了编号，编号的慕=目的是什么？
# 为了达到最后的txt文件，我们需要一步步构建路径
# HW3/training/底下是各个种类名称命名的文件夹，所以会根据名称在构建路径
# 又因为每一个种类的文件夹里面拥有大量的txt文档，所以最后的路径是categery_path + fileaname
# 为什么会有编号？因为计算机只认识数字，而编号的意义就在于将每一个分类apply到一个数字上，可以提出质疑的是，会不会因为数字的大小而对分类造成影响

# test_list = []

# for filename in os.dirname(TESTING_PATH):
# 	file_path = TESTING_PATH + filename

# 	with open(file_path, encoding = 'utf-8') as file:
# 		words = file.read().strip().split('/')
# 		test_list = append([words])

test_list = []

for i in range(1000):
	file_path = TESTING_PATH + str(i) + '.txt'

	with open(file_path, encoding = 'utf-8') as file:
		words = file.read().strip().split('/')
		test_list.append([i,words])

# 学习：
# read()函数的使用：file直接调用，另外是pandas内置函数read_pickle()
# 以及append()函数的使用针对list
# 这里利用i的目的是什么？


# 将处理好的资料转化为DataFrame
test_df = pd.DataFrame(test_list, columns = ["id", "text"])
print("Shape:" , test_df.shape)
print(test_df.sample(5))
#转化为pickle文件
test_df.to_pickle('test.pkl')
test_pickle_df=pd.read_pickle('test.pkl')
print(test_df.equals(test_pickle_df))


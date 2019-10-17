#practice: linear Rgression

import tensorflow as tf
import numpy as np


#creating training data,多数时候是直接从文件读入，需要注意的是，这只是训练数据，接下来我们需要去构建graph
trainX = np.linspace(-1,1,101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

#对tensor的定义,一个细节要注意，就是对于变量的定义Variable首字母要大写

#firstly, the setting pf placeholders
#X接受来自文件的真实数据
#Y同样也接受来自数据源的数据
#这也是placehole存在的理由，就是让源数据在grapg里面“安家”
X = tf.placeholder("float")
Y = tf.placeholder("float")
#接下来，我们需要去构建这种我们需要去拟合的关系
#因为我们是线性拟合，所以我们首先定义weighht
w=tf.Variable(0.0 , name = "weight")

#对Node的定义，即对operation的定义

#给出关系式
#y_model = tf.Variable(X*w)
#这是2018.4.21第一次练习时的模样，错误在于没有对tensorflow理解的很透彻；
y_model = tf.multiply(X,w)
#上述是该有的样子，之所以说没有对tensorflow很好的理解，是因为我们此时需要定义的是一个operation，是一个Node，而不是一个tensor，这很重要

#那么，我们利用w作为权重计算出的值与真实值相差多少，我们需要不懈努力，使得误差变得最小
loss_function = tf.pow(Y-y_model,2)

#接下来，是我们定义如何进行训练
train_operation = tf.train.GradientDescentOptimizer(0.01).minimize(loss_function)

#最后，我们刻画训练的过程

#首先，我们定义初始化函数，该函数也属于tensor
initialize_funtion = tf.global_variables_initializer()
#下一步就是让graph开始flow起来
with tf.Session() as sess:
	sess.run(initialize_funtion)
	for i in range(100):
		for (x,y) in zip(trainX, trainY):
			sess.run(train_operation,feed_dict={X: x,Y: y})
			print(sess.run(w))
	print(sess.run(w))
#从原始数据里面取出数据“入坑”，x->X,y->Y,Y被用来在loss_function中使用
#切记：feed_dict是一个类似于字典类型的关键值对，所以切记用中括号{}

#Please note that:
#first thing that has been done is to initialize the variables by calling init inside session.run().
#Later we run train_op by feeding feed_dict.
#Finally, we print the value of w(again inside sess.run() which should be around 3.

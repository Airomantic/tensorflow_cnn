import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot
# %matplotlib inline #只用于jupyter notebook，用于显示图像到该网页上，其它平台用plt.show()就可以显示绘制的图像了
import numpy as np
#print(tf.config.list_physical_devices('GPU'))
fashion_mnist=keras.datasets.fashion_mnist
#参与数据集
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()  #注意load_data()方法

print(train_images.shape) #60000张数据，28*28的
print(train_labels.shape) #(60000,)
print(test_images.shape) #(10000, 28, 28)
print(test_labels) #数组 [9 2 1 ... 8 1 5]
#认识图像：高 宽 通道数（RGB）
train_images=np.expand_dims(train_images,-1) #扩展成 四维数组
#数据形状(None,hight,width,chanal) 个数，高，宽，管道 以一张图片的形式输入进去，不会扁平化
print(train_images.shape) #(60000, 28, 28, 1) 即60000*28*28*1 chanal=1为黑白 chanal=3彩色图片RGB
#搭建卷积神经网络模型
model=tf.keras.Sequential() #顺序模型
#添加-卷积层 第一层使用卷积层（因为其特征提取能力远远大于全连接层）
#input_shape输入图片的形状，[1:]表示1->28，第一个像素值到最后一个，这里的train_images.shape[1:]为(28,28,1)
#图片28*28当用32个卷积核它的最后一位chanal会变厚变成32，
#然后默认用padding: str = 'valid'这种填充方式，则结果model.output_shape=(None, 26, 26, 32)
#padding='same'时model.output_shape=(None, 28, 28, 32)
model.add(tf.keras.layers.Conv2D(32,(3,3),
                                 input_shape=train_images.shape[1:],
                                 activation='relu',
                                 padding='same'))

#最大池化层
# model.output_shape=(None, 14, 14, 32)
model.add(tf.keras.layers.MaxPooling2D()) #没添加参数，默认 pool_size: Tuple[int, int] = (2, 2) 变成原来一半14*14
#卷积 (None, 12, 12, 64)
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu')) #超算数选择中的经验方法，使用2^n的卷积核个数，这样设计出的拟合能力很强，卷积核尺寸大小3*3
#以上四维形状都不能直接与输出（全连接）相连接，全连接层接受的是二维数据，所以需要把四维数据扁平化，添加一个GlobalAvgPool2D层，在所有的维度上做了一个平均
#全局平均池化
model.add(tf.keras.layers.GlobalAvgPool2D()) #(None, 64) 即在卷积层(None, 12, 12, 64)中12*12做一个平均得到64这么一个维度
#全连接层Dense
model.add(tf.keras.layers.Dense(10,activation='softmax')) #得到一个概率值 (None, 10)
#print(model.output_shape)
#print(model.summary())

#训练前首先配置
model.compile(optimizer='adam', #优化函数
              loss='sparse_categorical_crossentropy', #因为使用的label是1 2 3这样的
              metrics=['acc'] #使训练过程中输出正确率
              )
'''
训练30个epochs，通过比较其中的迭代 次数 （ Epochs ），误差表现（Performance）这些参数可知，
当隐含层神经元个数为10时，网络训练的误差最低，此时经过60迭代达 到训练目标。
​'''
history=model.fit(train_images,train_labels,
                  epochs=30, #训练30个epochs
                  validation_data=(test_images,test_labels) #可以看到训练过程中在validation_data上的训练情况
                  )
print(history)
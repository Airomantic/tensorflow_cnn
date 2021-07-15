
import tensorflow as tf
#print('vesion:{}'.format(tf.__version__))

from tensorflow import keras #该版本tensorflow内部集成了keras，并不是本身keras这个库
import matplotlib.pyplot as plt
import numpy
import glob #编写路径表达式，直接获取符合表达式的文件，python标准库，无需手动安装

all_image_path=glob.glob('2_class/*/*.jpg')

#对输入的数据做乱序
import random
random.shuffle(all_image_path)
#print(all_image_path[:5]) #用切片返回前五张
#print(all_image_path[-5:]) #后五张
#创建列表表示一类
label_to_index={'airplane':0,"lake":1}
#k,v反转，将0，1翻译成真实的值，如0表示airplane，1表示lake
index_to_label=dict((v,k)for k,v in label_to_index.items())
print(index_to_label)
img=all_image_path[100]
#mac linux里面的分割符可能与windows不一样
print(img.split('/')[1])#因为是[0,1,2]对应1，2，3 取2_class/lake/lake_136.jpg 里面的第2项
print(label_to_index.get(img.split('/')[1])) #得到1
#列表推导式，将所有的图片路径转换成all_labels 对所有图片路径进行迭代
all_labels=[label_to_index.get(img.split('/')[1]) for img in all_image_path] #取出所有路径标签
#print(all_labels)
#读取图片
img_raw=tf.io.read_file(img) #得到二进制形式
# print(img_raw)
#解码 选择.jpeg格式去解码二进制，之后变成图片的tensor，
img_tensor=tf.image.decode_jpeg(img_raw) #默认channels=0
#print(img_tensor.shape) #(256, 256, 3) 长 宽 管道RGB
#print(img_tensor.dtype) #数据类型<dtype: 'uint8'> 即0～255
#转换数据类型
img_tensor=tf.cast(img_tensor,tf.float32)
#我们希望数据归一化 方法：除以255 =>从0~1
img_tensor=img_tensor/255
#print(img_tensor) #([[[[[...]]]]],shape=(256, 256, 3), dtype=float32)
#print("numpy:",img_tensor.numpy()) #取出ndl类型，即去掉shape=(256, 256, 3), dtype=float32这两个参数
print("numpy:",img_tensor.numpy().max())
#把上面的步骤封装到一个函数里
def load_img(path):
    #读取图片
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3) #彩色图片，明确为3
    #加入输入的图片是不同大小的，需改变图片大小，规范化为同一大小，虽然会扭曲，但不会改变图片内容
    #tf.data 本质是一个输入管道，图片的大小多大都可以（200*200,300*300都可通过），如果不去规划好大小，管道会反馈回来-unkown不知道这张图片的大小
    img_tensor=tf.image.resize(img_tensor,[256,256])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255
    return img_tensor

i=random.choice(range(len(all_image_path))) #随机选取一张图片
print(i)
img_path=all_image_path[i]
label=all_labels[i]
#调用函数转换成Tensor
img_tensor=load_img(img_path)
#如果要绘图，需要将tensor转化numpy再绘图
plt.title(index_to_label.get(label))
plt.imshow(img_tensor.numpy())
plt.show()
#创建数据集
img_ds=tf.data.Dataset.from_tensor_slices(all_image_path)
#print(img_ds) #<TensorSliceDataset shapes: (), types: tf.string>
img_ds=img_ds.map(load_img) #对序列中的每一个元素都应用map()，把加载函数放入
#出现<MapDataset shapes: (256, 256, 3), types: tf.float32>
print(img_ds)
label_ds=tf.data.Dataset.from_tensor_slices(all_labels)
print(label_ds) #TensorSliceDataset shapes: ()空的表一个元素，单个的标量值
for la in label_ds.take(10):
    print(la.numpy()) #只显示0，1
    #print(index_to_label.get(la.numpy()))
#化成一个img_labels数据集，划分训练数据和测试数据
#将img和label合并成一个dataset
img_label_ds=tf.data.Dataset.zip((img_ds,label_ds)) #以元祖序列的形式放进去的，所以需要两层括号
image_count=len(all_image_path) #总的图片参数
#可选取其中80%作为训练，20%作为测试
test_count=int(image_count*0.2) #有可能是非整数，需要加个int()
train_count=image_count-test_count
train_ds=img_label_ds.skip(test_count) #跳过20%的test，剩下的都作为train
test_ds=img_label_ds.take(test_count) #take取前面的
#让他重复，希望它乱序，如果是固定顺序，那模型学习就会按顺序来学习，每次的学习得到的能力就一样，没有提高
#因为训练是一个batch一个batch的训练，如果把整个文件直接放进去，16G内存会溢出
#小批量的梯度下降
BATCH_SIZE=16
#当管道中的这个数据完毕之后，它就会重复的输出数据，这样的话dataset就可以不停的输出数据进行训练
#如果没有这个repeat()，当train_ds迭代一次之后就减速来
#shuffle()设置多大的缓存范围内，对这个数据（图片）进行乱序，在一定范围内缓存越大越好，但缓存大的超过一定范围，就都需要放到内存中，导致内存爆炸
#产生缓存区，对100张图片进行缓存
train_ds=train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
#<BatchDataset shapes: ((None, 256, 256, 3), (None,)), types: (tf.float32, tf.int32)>
#None是第一维度是batchv，则一个BATCH_SIZE就是16*256*256*3，增加了一个维度，所以不能重复运行batch()，每运行一次会增加一个维度None(出现1个None，2个None...)
print(train_ds)
#test数据没必要做乱序，结果一样，也不要repeat()，因为每一次内部训练的机制对train数据训练完之后，就将test数据只测试一遍
test_ds=test_ds.batch(BATCH_SIZE)

#模型的创建
#卷积网络来对图像进行识别
model=tf.keras.Sequential() #顺序模型
#卷积层 ，卷积核的个数，大小，每一个卷积核扫过图片都会形成一个特征层，即64个卷积核扫描过后形成64个维度的特征层，变厚channel=64
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation='relu'))
#这一层就不需要告诉输入形状来
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
#池化层 长宽减半，减少参数 MaxPool2D()和MaxPooling2D()效果是一样的
model.add(tf.keras.layers.MaxPool2D())
#重复 依次卷积核翻倍 模型拟合能力增强
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
#一直是四维 batch hight width channel 每经过一次Pool2D层减半（除以2，图像按2*2尺寸减少）图像大小：64->32->16->8->4->2
#全连接层 二分类
#全局平均池化，在hight width channel这个三个维度上求平均
model.add(tf.keras.layers.GlobalAveragePooling2D())
#Dense层 连接1024个单元数
model.add(tf.keras.layers.Dense(1024,activation='relu'))
#要输出来，可适当的减小单元数
model.add(tf.keras.layers.Dense(256,activation='relu'))
#二分类问题 逻辑回归
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
#输出单个的值，如果是>=1，则是分类为1这个类，如果1>''>0，则属于0这一类
model.summary()

# 可以在Sequential()中以列表的形式将这些层添加进去
# model2=tf.keras.Sequential([
#     tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),
#                            activation='relu'),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPool2D()
# ])
#编译模型，配置模型-优化器，损失函数，训练指标
#optimizer='adam'会设置默认学习速度0.01
#tf.keras.optimizers.Adam() 推荐设置小的学习速率，如果设置大的容易引起梯度的震荡（学习的正确率不能上升）
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
#所有的损失函数都在这个lossess，这里选择BinaryCrossentropy是为了应付分流问题
              #binary_crossentropy这个小写函数，pred值和实际的值作为参数传进去，而不能实现调用
              #当用大写时，返回的是个损失函数对象，是可调用
              #CategoricalCrossentropy 多分类问题,如果Dense(1,activation='sigmoid')输出做了激活，则from_logits: bool = False，如果未激活就要设置为true
              loss=tf.keras.losses.CategoricalCrossentropy(), #模型会调用loss，把它看出一个函数(输入一下损失值)
              metrics=['acc']
              )
#训练之前先规定两个参数
#每一个epoch表示将所有数据训练一遍，epoch就是train_count个数据除以batch尺寸
steps_per_epoch=train_count//BATCH_SIZE
#训练多少个Batch数据会将test数据训练一遍
validation_steps=test_count//BATCH_SIZE
history=model.fit(train_ds,epochs=5,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=test_ds,
                  validation_steps=validation_steps #训练多少个batch等于epoch
                  )
history.history.keys()
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
#plt.show()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
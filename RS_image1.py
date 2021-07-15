import tensorflow as tf
import pandas as pd
import keras
from keras import layers

#卷积层 （层都是大写的）图像一般都是用Conv2D来卷积的（二维数据）
#filters: Any, 卷积核的个数，对应通道个数
#kernel_size: Any （卷积核大小 3*3或5*5 等等）
#strides: int = (1, 1) 跨度，在x,y两个方向上移动，如果是（2，2）的话，就跳过一个，图像会减小
#padding: str = 'valid', 填充，valid表不填充，尽量使卷积核在这个像素里面移动，如果最后不能被5整除，则生成的像素就会有一点小
#use_bias: bool = True,是否添加权重（训练的值），它和图像上对应的像素值相乘然后相加
#kernel_initializer: str = 'glorot_uniform', 内核初始化，glorot_uniform（通用）会考虑输入输出
#kernel_regularizer: Any = None, 正则化，计算卷积时也会用到，比如把l2的权重（网络大小）作为loss的一部分
layers.Conv2D()
#池化层 下采样downsampling 如一张图像是3000*3000像素，非常大，要一步步把它降小
#最大池化：选取池化核（如2*2），有点类似卷积核，它会选取这个4个像素值中最大的值，然后 移动到下一个2*2（没有重叠）再选取最大
#strides: Any = None, 一般不会在池化的适合跳跃某些像素，导致有效信息减少
layers.MaxPooling2D()
#全连接层 前面提取的所有特征都用到一起，最后得到一个判断。不会使用一部分做判断
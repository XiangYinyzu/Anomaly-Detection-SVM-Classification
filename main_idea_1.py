#####改进TCN的对海雷达目标检测
from tensorflow.keras.models import Model        # 导入模型类，用于构建神经网络模型
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, LayerNormalization, Dropout,Attention,Layer,Add  # 导入各种神经网络层
from tensorflow.keras.utils import to_categorical   # 用于将标签转换为 one-hot 编码的工具函数
import scipy.io  # 用于读取 MATLAB 文件的工具库
                # 用于处理各种文件格式的输入输出操作。具体来说，它提供了读取和写入 MATLAB 文件（.mat 文件）的功能，以及其他一些常见文件格式的支持
                #与 MATLAB 文件相关的主要函数包括：scipy.io.loadmat()：用于读取 MATLAB 文件（.mat 文件）中的数据并返回为 Python 对象。
                                            #scipy.io.savemat()：用于将 Python 对象保存为 MATLAB 文件（.mat 文件）
import numpy as np   # 用于处理数值计算的库
import time    # 用于计算代码执行时间的模块
from sklearn.model_selection import train_test_split    # 用于划分训练集和测试集的工具函数
from tensorflow.keras.regularizers import l2  # 导入 L2 正则化器，用于在模型训练中减小过拟合
import tensorflow as tf  # 导入 TensorFlow 库，提供深度学习框架的功能

sequence_length = 1024         #每个序列数据的长度
##获取目标序列数据
def sequence_data1(path):
    # 使用scipy.io.loadmat函数加载.mat文件
    data1 = scipy.io.loadmat(path)   # 用于读取.mat文件中的数据并返回为 Python 对象
    Data1 = data1['AM']  #从data1 的 Python 字典中提取键为 'AM' 的项，并将其赋值给变量 Data1。
    #print(Data1)
    Data1 = Data1[0]     #将变量 Data1 中的第一个元素重新赋值给 Data1
    #print(Data1)
    # 指定n，即要组合的元素个数
    n = sequence_length  # 代表序列数据的长度，n=1024
    # 使用列表推导生成划分后的列表
    DATA1 = [Data1[i:i + n] for i in range(0, len(Data1)-1024, 10)]    #将 Data1 中的数据按照长度为 n 的滑动窗口进行切片，并存储在列表 DATA1 中。
                                                           #Data1[i:i + n] 表示从 Data1 中取索引从 i 开始、长度为 n 的子列表
                                                           #range(0, len(Data1)-1024, 10) 则表示从索引 0 开始，以步长为 10 逐渐增加直至 len(Data1)-1024（即 Data1 的长度减去 1024）
    # print(DATA1[1])
    return DATA1
#打标签
def Makelabel(DATA1,label):    # # 定义函数 Makelabel，接受两个参数：DATA1 作为输入数据，label 作为对应的标签
    if label == 1:
        labels = np.ones((DATA1.shape[0], 1))  # 创建一个形状为 (DATA1.shape[0], 1) 的全为 1 的数组       DATA1.shape[0]为
    else:
        labels = np.zeros((DATA1.shape[0], 1))     # 创建一个形状为 (DATA1.shape[0], 1) 的全为 0 的数组
    # print(label.shape)
    sdata = np.concatenate((DATA1, labels), axis=1)   #np.concatenate() 函数接受一个元组作为第一个参数，该元组包含需要拼接的数组，以及一个 axis 参数指定拼接的方向
                                                      #   axis=1 表示沿着列方向（即水平方向）进行拼接
                                                      ## 将数据 DATA1 和标签 labels 沿着列方向拼接在一起，形成新的数组 sdata
    # print(sdata.shape)
    return sdata

#读取划分数据并打标签，获取标记好的目标序列数据
def process_dataset1(path1, labels):   # 定义函数 process_dataset1，接受两个参数：path1，labels

    DATA = sequence_data1(path1)    #数据文件的路径path1输入sequence_data1函数中，经过数据读取和划分，输出目标序列数据DATA       将整个数据单元按一定长度划分序列
    #print(DATA)
    DATA = np.array(DATA)      #这个函数的主要作用是将非 NumPy 数据结构转换为 NumPy 数组，以便进行更方便的数值计算和数据处理。
    # print(DATA.shape)
    DATA = DATA.reshape((-1, sequence_length))    # 使用 reshape 函数将数组 DATA 重新塑形为指定形状，
                                            # 其中 -1 表示自动计算对应维度的大小，sequence_length 表示每个子数组的长度
                                            # 这将使得 DATA 被重新塑形为一个二维数组，每个子数组的长度为 sequence_length
    # print(DATA)
    # print(DATA.shape)
    # DATA = np.array(DATA[:int(131072/name)])   #序列长度不够10的删除，并转为numpy
    # #print(DATA.shape)
    # DATA = DATA.reshape((int(131072/name), name))
    # 打标签
    data1 = Makelabel(DATA, labels)      #将划分好的序列数据DATA和所需标签labels输入 到  Makelabel函数中，最终输出标记好的序列数据
    # print(len(trainData))
    # print(len(testData))
    #eeeeeeeeeeeeeeee
    return data1

##获取海杂波序列数据
def sequence_data(path):
    # 使用scipy.io.loadmat函数加载.mat文件
    data1 = scipy.io.loadmat(path)  # 用于读取.mat文件中的数据并返回为 Python 对象
    Data1 = data1['AM']    #从data1 的 Python 字典中提取键为 'AM' 的项，并将其赋值给变量 Data1。
    #print(Data1)
    Data1 = Data1[0]   #将变量 Data1 中的第一个元素重新赋值给 Data1
    #print(Data1)
    # 指定n，即要组合的元素个数
    n = sequence_length  #代表序列数据的长度，n=1024
    # 使用列表推导生成划分后的列表
    # DATA1 = [Data1[i:i + n] for i in range(0, len(Data1)-1024, 10)]   #将原始数据 Data1 切分为长度为n=1024的子序列，并存储在列表 DATA1 中
    DATA1 = [Data1[i:i + n] for i in range(0, len(Data1), n)]
    # print(DATA1[1])
    # ddddddd
    return DATA1
# #读取划分数据并打标签，获取标记好的海杂波序列数据
def process_dataset(path, labels):   # 定义函数 process_dataset，接受两个参数：path，labels
    data = path
    DATA = sequence_data(data)  #数据路径path输入sequence_data函数中，经过数据读取和划分，输出目标序列数据DATA
    # print(DATA)
    DATA = np.array(DATA)  # 将变量 DATA 转换为 NumPy 数组，并将结果重新赋值给变量 DATA
    # print(DATA.shape)
    DATA = DATA.reshape((-1, sequence_length))   # 将DATA重新塑形为一个二维数组，每个子数组的长度为 sequence_length
    # print(DATA.shape)
    # 打标签
    data1 = Makelabel(DATA, labels)  #将数据DATA和标签labels输入 到  Makelabel函数中，最终输出标记好的海杂波序列数据
    return data1

# def multi_category_focal_loss1(gamma=2., alpha=.25):
#     """
#     focal loss for multi category of multi label problem
#     适用于多分类或多标签问题的focal loss
#     alpha控制真值y_true为1/0时的权重
#         1的权重为alpha, 0的权重为1-alpha
#     当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
#     当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
#     当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
#         尝试将alpha调大,鼓励模型进行预测出1。
#     Usage:
#      model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
#     epsilon = 1.e-7
#     gamma = float(gamma)
#     alpha = tf.constant(alpha, dtype=tf.float32)
#
#     def multi_category_focal_loss2_fixed(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
#
#         alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
#         y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
#         ce = -tf.math.log(y_t)
#         weight = tf.pow(tf.subtract(1., y_t), gamma)
#         fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
#         loss = tf.reduce_mean(fl)
#         return loss
#
#     return multi_category_focal_loss2_fixed

#自注意力机制
class SelfAttention(Layer):
    #Python 类 SelfAttention 的构造函数 __init__()，用于初始化该类的实例
    def __init__(self, units):     #构造函数的定义，它接受一个参数 units，用于指定自注意力机制中的单位数或维度。
        """
                Initializes the SelfAttention layer.
                Args:
                    units (int): The number of units or dimensions for the attention mechanism.
                """
        super(SelfAttention, self).__init__()   #这是调用父类 Layer 的构造函数，用于确保正确地初始化父类中的属性。
        self.units = units    #这一行代码将传入的 units 参数赋值给类的属性  self.units：在类的其他方法中可以使用这个属性来访问自注意力机制中的单位数或维度。

    #SelfAttention 类中的 build 方法，用于构建该层的权重
    ####在构建自注意力层时，初始化了该层的权重矩阵和偏置项，并使用了 L2 正则化器和 Xavier/Glorot 初始化器来提高模型的泛化能力和训练稳定性。
    def build(self, input_shape):   #这是方法的定义，它接受一个参数 input_shape，表示输入张量的形状。
        """
                Builds the weights of the SelfAttention layer.
                Args:
                    input_shape (tuple): The shape of the input tensor.
                Returns:
                    None
                """
        # 在权重初始化时添加L2正则化项        添加正则化项和可选的偏差项（bias）以提高模型的泛化能力。
        kernel_regularizer = tf.keras.regularizers.l2(0.01)  #定义了一个 L2 正则化器，用于在权重初始化时添加 L2 正则化项，以减小过拟合风险
        initializer = tf.initializers.GlorotUniform()    #使用Xavier/Glorot初始化来改进权重矩阵的初始化。这通常可以提高训练的稳定性和速度。
        #这些权重矩阵和偏置项将在自注意力机制中用于计算查询、键和值。
        # 初始化权重矩阵 W_q, W_k, W_v
        self.W_q = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer=initializer,
                                   trainable=True,
                                   regularizer=kernel_regularizer,
                                   name = 'W_q')   #self.add_weight() 方法用于添加模型的可训练权重。
                                                  # 在这里，它为权重矩阵和偏置项分别指定了形状、初始化方式、是否可训练、正则化器和名称等参数。
        self.W_k = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer=initializer,
                                   trainable=True,
                                   regularizer=kernel_regularizer,
                                   name = 'W_k')
        self.W_v = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer=initializer,
                                   trainable=True,
                                   regularizer=kernel_regularizer,
                                   name = 'W_v')

        # # 初始化偏差项 bias_q, bias_k, bias_v
        self.bias_q = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True,
                                      name = 'bias_q')
        self.bias_k = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True,
                                      name = 'bias_k')
        self.bias_v = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True,
                                      name = 'bias_v')
#定义了 SelfAttention 类中的 call 方法，用于计算自注意力机制
    #call 方法返回了应用自注意力机制后的输出张量 output，其形状与输入张量相同
    def call(self, inputs):  #这是方法的定义，它接受一个参数 inputs，表示输入张量
        """
               Computes the self-attention mechanism.
               Args:
                   inputs (tf.Tensor): The input tensor.

               Returns:
                   tf.Tensor: The output tensor after applying self-attention.
               """
        # 计算 Q，K，V  计算了输入张量 inputs 与权重矩阵 W_q、W_k、W_v 的乘积，得到了查询（q）、键（k）和值（v）。
        q = tf.matmul(inputs, self.W_q)   #tf.matmul(, ) 的作用是执行矩阵相乘运算,将输入数据映射到一个查询空间
                                          #结果是一个查询矩阵，其中每一行代表一个查询向量，用于计算注意力分数。
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)
        # 计算注意力分数和注意力权重
        attention_scores = tf.matmul(q, k, transpose_b=True)  #计算注意力分数，即查询与键的点积。
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  #计算注意力权重，通过对注意力分数进行 softmax 操作，将得分转换为概率。
        # 计算输出
        output = tf.matmul(attention_weights, v)  #使用注意力权重加权平均值计算输出，即注意力加权后的值。
        return output
    #用于将自定义层的配置信息保存到字典中。在序列化模型时，可以使用该配置信息来重新构建相同的自定义层。
    def get_config(self):
        config = super().get_config().copy()       #通过调用 super().get_config() 获取父类的配置信息，然后更新其中的自定义配置
        config.update({
            'units':self.units,
        })                              #这里是更新了自注意力层中的 units 参数。
        return config             #最后，返回包含更新配置的字典。


# 残差块函数
def ResBlock(x, filters, kernel_size, dilation_rate):      #输入数据x ,filters代表输出空间的维度（即输出特征的数量）,
                                                        # kernel_size卷积核的大小,dilation_rate卷积核的膨胀系数
    #第一次卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    r = LayerNormalization()(r)  # 层归一化
    r = Activation('relu')(r)  # 激活函数
    r = Dropout(rate=0.2)(r)  # Dropout 正则化
    #print(r.shape)
    # 第二次卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)
    r = LayerNormalization()(r)  # 层归一化
    r = Activation('relu')(r)  # 激活函数
    r = Dropout(rate=0.2)(r)  # Dropout 正则化
    ###创建残差连接中的shortcut，确保残差连接中的两条路径具有相同的维度，使得可以对它们进行元素级的相加操作。
    if x.shape[-1] == filters:    ###第一个卷积层输入数据的形状是（64，1024，1），与经过残差块后的输出数据维度不一样，
                                     # 所以创建残差连接中的shortcut，确保残差连接中的两条路径具有相同的维度
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)
    shortcut = LayerNormalization()(shortcut)

    o = add([r, shortcut])
    o = Activation('relu')(o)  # 激活函数
    o = Dropout(rate=0.2)(o)
    return o

# 序列模型
def TCN(input_shape):
    #print(input_shape)
    inputs = Input(shape=input_shape)         #input_shape 是用来描述单个数据序列的形状，而 inputs 张量则包含了整个训练批次的数据形状，包括批次大小。
                                              #Input:创建模型输入层的函数    输入：(1024,1)            输出：(64,1024,1)
    #print(inputs)
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)  #输入：(64,1024,1)   输出：(64,1024,64)
    x = tf.keras.layers.MaxPooling1D(4)(x)  #输入：(64,1024,64)   输出：(64,256,64)
    x = Dropout(0.2)(x)    #输入：(64,256,64)   输出：(64,256,64)

    attention_Output = []
    num_filters = 32
    #print('1111111111111111111')
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=1)  #输入：(64,256,64)   输出：(64,256,32)
    #print(x)
    # 应用自注意力层
    y1 = SelfAttention(units=num_filters)(x)
    attention_Output.append(y1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)   #输入：(64,256,32)   输出：(64,256,32)
    # 应用自注意力层
    y2 = SelfAttention(units=num_filters)(x)
    attention_Output.append(y2)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=4)   #输入：(64,256,32)   输出：(64,256,32)
    # 应用自注意力层
    y3 = SelfAttention(units=num_filters)(x)
    attention_Output.append(y3)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=8)   #输入：(64,256,32)   输出：(64,256,32)
    # print(attention_output)
    # 应用自注意力层
    y4 = SelfAttention(units=num_filters)(x)
    attention_Output.append(y4)
    # 将所有注意力层的输出合并
    x = Add()(attention_Output)    #输出：(64,256,32)
    print(x)
    x = Activation('relu')(x)    #输出：(64,256,32)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)      # 全局平均池化层  输出：(64,32)
    # ##全连接
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)  # 输出层，使用 softmax 激活函数进行分类    输出：(64,2)
    #print(x)
    model = Model(inputs=inputs, outputs=x)
    return model

#将每个单元划分为训练集和测试集
# Data_1 = process_dataset1('./AM 01 HH 1968.mat', 1)    #文件名AM 17 HH 01.mat中 AM：代表幅度  17：代表#17号数据集  HH：代表极化方式  01：0代表海杂波，1代表第1个距离单元
                                                       #文件名AM 17 HH 19.mat中 AM：代表幅度  17：代表#17号数据集  HH：代表极化方式  19：1代表目标，9代表第9个距离单元
# Data_01 = process_dataset1('./AM 01 HH 1969.mat', 1)
# Data_02 = process_dataset('./AM 01 HH 0967.mat', 0)
# Data_03 = process_dataset('./AM 01 HH 0970.mat', 0)
# Data_21 = process_dataset1('./AM 01 HH 1968.mat', 1)
# Data_021 = process_dataset1('./AM 01 HH 1969.mat', 1)
# Data_022 = process_dataset('./AM 01 HH 0967.mat', 0)
# Data_023 = process_dataset('./AM 01 HH 0970.mat', 0)
# Data_31 = process_dataset1('./AM 01 HH 1968.mat', 1)
# Data_031 = process_dataset1('./AM 01 HH 1969.mat', 1)
# Data_032 = process_dataset('./AM 01 HH 0967.mat', 0)
# Data_033 = process_dataset('./AM 01 HH 0970.mat', 0)
#将每个单元划分为训练集和测试集
Data_1 = process_dataset1('./AM 17 HH 19.mat', 1)
Data_01 = process_dataset('./AM 17 HH 01.mat', 0)
Data_02 = process_dataset('./AM 17 HH 02.mat', 0)
Data_03 = process_dataset('./AM 17 HH 03.mat', 0)
Data_04 = process_dataset('./AM 17 HH 04.mat', 0)
Data_05 = process_dataset('./AM 17 HH 05.mat', 0)
Data_09 = process_dataset('./AM 17 HH 06.mat', 0)
Data_010 = process_dataset('./AM 17 HH 07.mat', 0)
# Data_011 = process_dataset('./AM 17 HH 011.mat', 0)
Data_012 = process_dataset('./AM 17 HH 012.mat', 0)
Data_013 = process_dataset('./AM 17 HH 013.mat', 0)
Data_014 = process_dataset('./AM 17 HH 014.mat', 0)
#将多个单元合并
# Dataset = np.concatenate((Data_1, Data_01,Data_02, Data_03, Data_21, Data_021,Data_022, Data_023, Data_31, Data_031,Data_032, Data_033),  axis=0)
#将多个单元合并
Dataset = np.concatenate((Data_1, Data_01,Data_02, Data_03, Data_04, Data_05, Data_09, Data_010,
                          Data_012, Data_013, Data_014), axis=0)

#划分训练集和测试集
trainDataset, testDataset = train_test_split(Dataset, test_size=0.3, random_state=8, shuffle=True)
# print(len(testDataset))
# # print(trainData)
# print(len(trainDataset))
# ddddddddddddddddddddd
#将训练集、测试集的数据和标签单独存放
train_x = trainDataset[:,:sequence_length]   #训练集数据
trainLabel = trainDataset[:,sequence_length:(sequence_length+1)]   #训练集标签
train_y = to_categorical(trainLabel, 2)  #将整数标签类型转为one-hot编码

test_x = testDataset[:,:sequence_length]     #测试集数据
testLabel = testDataset[:,sequence_length:(sequence_length+1)]     ##测试集标签
test_y = to_categorical(testLabel, 2)
#print(trainData.shape,trainLabel.shape)

#print(trainData.shape)
#初始化模型
model = TCN((sequence_length, 1))    #sequence_length：输入数据序列的长度   1：输入数据序列的维度
# 查看网络结构
model.summary()
# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss=[multi_category_focal_loss1(alpha=.25, gamma=2)], metrics=['accuracy'])
# 训练模型
model.fit(train_x, train_y, batch_size=64, validation_split = 0.2, epochs=2, verbose=2)
# # 评估模型
# pre = model.evaluate(test_x, test_y, batch_size=16, verbose=2)
# print('test_loss:', pre[0], '- test_acc:', pre[1])
model.save_weights('17HH_weight.h5')

#####虚警率控制
####训练集控虚警
train_acc = model.predict(train_x)    #训练数据集中样本预测成杂波和目标的概率
print('1111111111111')
print(train_acc)
t_c = []    #建立一个空列表存放训练及中海杂波的预测概率
for i in range(len(train_acc)):
    if train_y[i][0] == 1:    #训练集中杂波样本
        t_c.append(train_acc[i][0])      #训练集中海杂波的预测概率
train_clutter_acc = sorted(t_c)        #从小到大排列
#print(train_clutter_acc)
cl = 0   #初始化训练集中海杂波数为0
ct = 0   #初始化训练集中目标数为0
for i in range(len(train_y)):
    if train_y[i][0] == 1:
        cl += 1   #训练集中海杂波数
    else:
        ct += 1    #训练集中目标数

PF = 0.001   #预设虚警率为0.001
w = PF * cl
T = train_clutter_acc[int(w)]   #阈值
PF1 = 0.01    #预设虚警率为0.01
w1 = PF1 * cl
T1 = train_clutter_acc[int(w1)]  # 阈值

c_a = 0  #预设虚警率为0.001时，训练集中初始化杂波判为目标的个数为0
c_b = 0   #0.001时，训练集中初始化目标判为目标的个数
c_a1 = 0  #0.01时，训练集中初始化杂波判为目标的个数为0
c_b1 = 0  #0.01时，训练集中初始化目标判为目标的个数
for i in range(len(train_acc)):
    if train_y[i][0] == 1:               #统计杂波判目标的个数
        if train_acc[i][0] <= T:
            c_a += 1
        if train_acc[i][0] <= T1:
            c_a1 += 1
    else:                              #目标判目标的个数
        if train_acc[i][0] <= T:
            c_b += 1
        if train_acc[i][0] <= T1:
            c_b1 += 1

########模型测试#######      通过 model.fit() 函数对模型进行了训练，训练过程中模型的权重已经被更新了。
                              #当模型训练完成后，模型的权重已经被保存在模型对象中。
                            # 因此，在测试阶段，只需简单地使用 model.predict(test_x) 来对测试集进行预测即可，模型会自动使用已经学到的权重进行预测。
test_acc = model.predict(test_x)   #测试集中样本预测为杂波和目标的预测概率

clutter = 0     #设测试集中海杂波数为0
target = 0       #设测试集中目标数为0
a = 0
b = 0
a1 = 0
b1 = 0
for i in range(len(test_acc)):
    if test_y[i][0] == 1:
        clutter += 1
        if test_acc[i][0] <= T:
            a += 1
        if test_acc[i][0] <= T1:
            a1 += 1
    else:
        target += 1
        if test_acc[i][0] <= T:
            b += 1
        if test_acc[i][0] <= T1:
            b1 += 1
print('0.001时训练的虚警和准确率分别为：', c_a/cl, c_b/ct)
print('0.001时测试的虚警和准确率分别为：', a/clutter, b/target)

print('0.01时训练的虚警和准确率分别为：', c_a1 / cl, c_b1 / ct)
print('0.01时测试的虚警和准确率分别为：', a1 / clutter, b1 / target)

print('%.3f, %.4f,%.2f, %.4f' %(c_a/cl, c_b/ct, c_a1/cl, c_b1/ct))
print('%.3f, %.4f,%.2f, %.4f' %(a/clutter, b/target,a1/clutter, b1/target))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import colors


# 绘制混淆矩阵
# 绘制混淆矩阵
test_pred = np.argmax(test_acc, axis=1)
test_true = np.argmax(test_y, axis=1)

cm = confusion_matrix(test_true, test_pred)

# 创建自定义颜色映射，使对角线颜色一致
cmap = colors.ListedColormap(['#FF9999', '#99FF99'])
bounds = [0, cm.max()/2, cm.max()]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap, norm=norm)

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")

# 在矩阵上标注数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# 设置标签和标题
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Clutter', 'Target'], yticklabels=['Clutter', 'Target'],
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

plt.show()
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow_1 import data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weight_variable(shape):
    # 产生随机变量
    # truncated_normal：选取位于正态分布均值=0.1附近的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 读取MNIST数据集
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 预定义输入值X、输出真实值Y    placeholder为占位符
x = tf.   placeholder(tf.float32, shape=[None, 1024], name='input_x')  # 1024个点
print(x)
y_ = tf.placeholder(tf.float32, shape=[None, 3], name='input_y')  # 3类   正常+外圈故障+内圈故障
keep_prob = tf.placeholder(tf.float32)
# 转换成二维数据
x_image = tf.reshape(x, [-1, 32, 32, 1])

print(x_image.shape)  # [n_samples,28,28,1]

# 卷积层1网络结构定义
# 卷积核1：patch=5×5;in size 1;out size 32;激活函数reLU非线性处理
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 32*32*32
h_pool1 = max_pool_2x2(h_conv1)  # output size 16*16*32#卷积层2网络结构定义

# 卷积核2：patch=5×5;in size 32;out size 64;激活函数reLU非线性处理
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 16*16*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 8 *8 *64

# 全连接层1
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])  # [n_samples,8,8,64]->>[n_samples,8*8*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 减少计算量dropout
print(h_fc1_drop.shape)
# 全连接层2
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(prediction)
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 二次代价函数:预测值与真实值的误差
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))

# 梯度下降法:数据太庞大,选用AdamOptimizer优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# saver = tf.train.Saver()  # defaults to saving all variables


# sess.run(tf.global_variables_initializer())

# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step", i, "training accuracy", train_accuracy)
#
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

def scale(x):
    # 标准化数据
    x = (x - 0.5) / 0.5
    return x


def show_acc(hist, batchsize, Epoch, show=False, save=False):
    x = range(len(hist))
    y1 = hist
    plt.plot(x, y1, label='acc')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend(loc=2)
    plt.grid(True)
    plt.tight_layout()
    if save:
        if not os.path.exists('../test_accuracy'):
            os.mkdir('../test_accuracy')
        plt.savefig("test_accuracy/Cnn_B{}E{}.png".format(batchsize, Epoch + 1))

    if show:
        plt.show()
    else:
        plt.close()


def train_CNN(batch_size, epochs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # mnist_data = get_data()
        # no_of_batches = int(mnist_data.train.images.shape[0] / batch_size) + 1
        no_of_batches = int(data.train_X.shape[0] / batch_size) + 1
        print("训练样本数量:")
        print(data.train_X.shape[0])
        test_acc = []
        for epoch in range(epochs):

            train_accuracies, train_losses = [], []

            for it in range(no_of_batches - 1):
                # batch = mnist_data.train.next_batch(batch_size, shuffle=False)
                # test_batch = mnist_data.train.next_batch(batch_size, shuffle=False)
                # batch[0] has shape: batch_size*28*28*1

                batch = data.train_X[it * batch_size:batch_size + it * batch_size, ]
                batch_label = data.train_Y[it * batch_size:batch_size + it * batch_size, ]
                # test_batch = mnist_data.train.next_batch(batch_size, shuffle=False)
                test_batch = data.test_X
                test_batch_label = data.test_Y

                # batch_reshaped = tf.image.resize_images(batch[0], [64, 64]).eval()
                # batch_reshaped = batch.reshape([-1, 32, 32, 1])
                test_batch_reshaped = test_batch.reshape([-1, 32, 32, 1])

                # Reshaping the whole batch into batch_size*64*64*1 for disc/gen architecture

                train_step.run(feed_dict={x: scale(batch), y_: batch_label, keep_prob: 0.5})
                test_feed_dict = {x: scale(test_batch), y_: test_batch_label, keep_prob: 1.0}
                # 字典中提供的是one—hot标签
                Loss = loss.eval(feed_dict={x: scale(batch), y_: batch_label, keep_prob: 0.5})
                train_accuracy = accuracy.eval(feed_dict=test_feed_dict)
                # test_prediction_type = prediction_type.numpy()
                train_losses.append(Loss)
                train_accuracies.append(train_accuracy)
                print('Epoch[{}]'.format(epoch), 'Batch evaluated [{}]/[{}]'.format(str(it + 1), no_of_batches - 1))
            print('开始计算loss')
            LOSS = np.mean(train_losses)
            tr_acc = np.mean(train_accuracies)
            print(train_accuracies)
            print(tr_acc)
            test_acc.append(tr_acc)
            print(test_acc)
            print('After epoch: ' + str(epoch + 1) + '   loss' + str(LOSS) + ' Accuracy: ' + str(tr_acc))
        show_acc(test_acc, batch_size, epoch, show=False, save=True)
        #  输出预测结果
        Fina_test_feed_dict = {x: scale(data.Fina_train_X), y_: data.Fina_train_Y, keep_prob: 1.0}
        prediction_type = tf.argmax(prediction, 1).eval(feed_dict=Fina_test_feed_dict)
        real_type = tf.argmax(y_, 1).eval(feed_dict=Fina_test_feed_dict)
        test_accuracy = accuracy.eval(feed_dict=Fina_test_feed_dict)
        print(test_accuracy)
        print(len(test_acc))
        dataw = open("D:\data.txt", 'w+')
        lenth = len(test_acc)
        for i in range(lenth):
            print(test_acc[i], file=dataw)
        dataw.close()
        # builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./Models')  # 保存前需要保证这个文件夹为空或者不存在
        # builder.add_meta_graph_and_variables(sess, [tf.saved_model.TRAINING])
        # builder.save()
        # saver.save(sess, 'checkpoints/Cnnmodel.ckpt')
        sess.close()
    return train_losses


train_CNN(50, 100)
# 保存模型参数
# saver.save(sess, './model.ckpt')

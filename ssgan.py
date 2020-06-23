# -*- coding: utf-8 -*-


# ########### 导包 ############

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True

# ########### 初始化 ############

num_classes = 3
channels = 1
height = 32
width = 32
# MNIST was resized to 64 * 64 for discriminator and generator architecture fitting
latent = 100
epsilon = 1e-7
labeled_rate = 0.2  # For initial testing


# ########### Importing MNIST data ########### #

def get_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=[])
    return mnist

    # ########### 标准化数据 ############


# 生成器生成的 tanh output 缩放到（-1.，1）
def scale(x):
    # 标准化数据
    x = (x - 0.5) / 0.5
    return x


"""判别器和生成器的结构是相互镜像的"""


# ########### 定义判别器（分类器） ############


def discriminator(x, dropout_rate=0., is_training=True, reuse=False):
    # input x -> n+1 classes
    # 输入数据，一共n+1类数据，最后一类数据为生成器生成的
    with tf.variable_scope('Discriminator', reuse=reuse):
        # x = ?*28*28*1
        x = tf.reshape(x, [-1, 32, 32, 1])
        print('Discriminator architecture: ')
        # Layer 1
        conv1 = tf.layers.conv2d(x, 128, kernel_size=[4, 4], strides=[1, 1],
                                 padding='same', activation=tf.nn.leaky_relu, name='conv1')  # ?*32*32*128
        print(conv1.shape)
        # No batch-norm for input layer
        dropout1 = tf.nn.dropout(conv1, dropout_rate)

        # Layer2
        conv2 = tf.layers.conv2d(dropout1, 256, kernel_size=[4, 4], strides=[2, 2],
                                 padding='same', activation=tf.nn.leaky_relu, name='conv2')  # ?*16*16*256
        batch2 = tf.layers.batch_normalization(conv2, training=is_training)
        dropout2 = tf.nn.dropout(batch2, dropout_rate)
        print

        # Layer3
        conv3 = tf.layers.conv2d(dropout2, 1024, kernel_size=[4, 4], strides=[4, 4],
                                 padding='same', activation=tf.nn.leaky_relu, name='conv3')  # ?*2*2*1024

        # batch3 = tf.layers.batch_normalization(conv3, training=is_training)
        # dropout3 = tf.nn.dropout(batch3, dropout_rate)
        print(conv3.shape)

        # Layer 4
        # conv4 = tf.layers.conv2d(dropout3, 1024, kernel_size=[3, 3], strides=[1, 1],
        #                        padding='valid', activation=tf.nn.leaky_relu, name='conv4')  # ?*2*2*1024
        # No batch-norm as this layer's op will be used in feature matching loss
        # No dropout as feature matching needs to be definite on logits
        # print(conv4.shape)

        # Layer 5
        # Note: Applying Global average pooling

        flatten = tf.reduce_mean(conv3, axis=[1, 2])
        logits_D = tf.layers.dense(flatten, (1 + num_classes))
        out_D = tf.nn.softmax(logits_D)

    return flatten, logits_D, out_D


# ########### 定义生成器 ############

def generator(z, dropout_rate=0., is_training=True, reuse=False):
    # input latent z -> image x

    with tf.variable_scope('Generator', reuse=reuse):
        print('\n Generator architecture: ')

        # Layer 1
        deconv1 = tf.layers.conv2d_transpose(z, 512, kernel_size=[4, 4],
                                             strides=[1, 1], padding='valid',
                                             activation=tf.nn.relu, name='deconv1')  # ?*4*4*512
        batch1 = tf.layers.batch_normalization(deconv1, training=is_training)
        dropout1 = tf.nn.dropout(batch1, dropout_rate)
        print(deconv1.shape)

        # Layer 2
        deconv2 = tf.layers.conv2d_transpose(dropout1, 256, kernel_size=[4, 4],
                                             strides=[4, 4], padding='same',
                                             activation=tf.nn.relu, name='deconv2')  # ?*16*16*256
        batch2 = tf.layers.batch_normalization(deconv2, training=is_training)
        dropout2 = tf.nn.dropout(batch2, dropout_rate)
        print(deconv2.shape)

        # Layer 3
        deconv3 = tf.layers.conv2d_transpose(dropout2, 128, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same',
                                             activation=tf.nn.relu, name='deconv3')  # ?*32*32*128
        batch3 = tf.layers.batch_normalization(deconv3, training=is_training)
        dropout3 = tf.nn.dropout(batch3, dropout_rate)
        print(deconv3.shape)

        # Output layer
        deconv4 = tf.layers.conv2d_transpose(dropout3, 1, kernel_size=[4, 4],
                                             strides=[2, 2], padding='same',
                                             activation=None, name='deconv4')  # ?*64*64*1
        yasuo = tf.image.resize(deconv4, [32, 32])  # ?*32*32*1
        out = tf.nn.tanh(yasuo)
        print(deconv4.shape)
        print(yasuo.shape)

    return out


# ########### 建模 ############

def build_GAN(x_real, z, dropout_rate, is_training):
    fake_images = generator(z, dropout_rate, is_training)

    D_real_features, D_real_logits, D_real_prob = discriminator(x_real, dropout_rate,
                                                                                  is_training)

    D_fake_features, D_fake_logits, D_fake_prob = discriminator(fake_images, dropout_rate,
                                                                                  is_training, reuse=True)
    # Setting reuse=True this time for using variables trained in real batch training

    return D_real_features, D_real_logits, D_real_prob, D_fake_features, D_fake_logits, D_fake_prob, fake_images


# ########### Preparing Mask ############

# Preparing a binary label_mask to be multiplied with real labels
# 准备与真实数据的标签相乘的二进制标签掩码
def get_labeled_mask(labeled_rate, batch_size):
    labeled_mask = np.zeros([batch_size], dtype=np.float32)
    labeled_count = np.int(batch_size * labeled_rate)
    labeled_mask[range(labeled_count)] = 1.0
    np.random.shuffle(labeled_mask)
    return labeled_mask


# ########### 扩展标签，加入第n+1类标签 ############

def prepare_extended_label(label):
    # add extra label for fake data
    extended_label = tf.concat([tf.zeros([tf.shape(label)[0], 1]), label], axis=1)

    return extended_label


# ########### 定义loss值 ############

# The total loss inculcates  D_L_Unsupervised + D_L_Supervised + G_feature_matching loss + G_R/F loss

def loss_accuracy(D_real_features, D_real_logit, D_real_prob, D_fake_features,
                  D_fake_logit, D_fake_prob, extended_label, labeled_mask, D_real_mid_logit, D_fake_mid_logit):
    # ## Discriminator loss ###

    # Supervised loss -> which class the real data belongs to

    temp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_real_logit,
                                                      labels=extended_label)
    # Don't confuse labeled_rate with labeled_mask
    # Labeled_mask and temp are of same size = batch_size where temp is softmax
    # cross_entropy calculated over whole batch

    D_L_Supervised = tf.reduce_sum(tf.multiply(temp, labeled_mask)) / tf.reduce_sum(labeled_mask)

    # Multiplying temp with labeled_mask gives supervised loss on labeled_mask
    # data only, calculating mean by dividing by no of labeled samples

    # Unsupervised loss -> R/F

    D_L_RealUnsupervised = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_logit[:, 0], labels=tf.zeros_like(D_real_logit[:, 0], dtype=tf.float32)))

    D_L_FakeUnsupervised = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logit[:, 0], labels=tf.ones_like(D_fake_logit[:, 0], dtype=tf.float32)))

    D_L = D_L_Supervised + D_L_RealUnsupervised + D_L_FakeUnsupervised

    # ## Generator loss ###

    # G_L_1 -> Fake data wanna be real

    G_L_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logit[:, 0], labels=tf.zeros_like(D_fake_logit[:, 0], dtype=tf.float32)))


    # G_L_2 -> Feature matching
    data_moments = tf.reduce_mean(D_real_features, axis=0)
    sample_moments = tf.reduce_mean(D_fake_features, axis=0)
    G_L_2 = tf.reduce_mean(tf.square(data_moments - sample_moments))

    G_L = G_L_1 + G_L_2
    prediction_value = tf.argmax(D_real_prob[:, 1:], 1)
    prediction = tf.equal(tf.argmax(D_real_prob[:, 1:], 1),
                          tf.argmax(extended_label[:, 1:], 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return D_L, G_L, accuracy, prediction_value


# ########### Defining Optimizer ############

def optimizer(D_Loss, G_Loss, learning_rate, beta1):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        all_vars = tf.trainable_variables()
        D_vars = [var for var in all_vars if var.name.startswith('Discriminator')]
        G_vars = [var for var in all_vars if var.name.startswith('Generator')]

        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1,
                                             name='d_optimiser').minimize(D_Loss, var_list=D_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1,
                                             name='g_optimiser').minimize(G_Loss, var_list=G_vars)

    return d_train_opt, g_train_opt


# ########### 画出结果 ############

def show_result(test_images, num_epoch, show=True, save=False):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i in range(0, size_figure_grid):
        for j in range(0, size_figure_grid):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (32, 32)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        if not os.path.exists('images'):
            os.mkdir('images')
        fig.savefig("images/%d.png" % num_epoch)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("images/Train_hist.png")

    if show:
        plt.show()
    else:
        plt.close()


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
        if not os.path.exists('test_accuracy'):
            os.mkdir('test_accuracy')
        plt.savefig("test_accuracy/B{}E{}.png".format(batchsize, Epoch + 1))

    if show:
        plt.show()
    else:
        plt.close()


# ########### TRAINING ############


def train_GAN(batch_size, epochs):
    train_hist = {'D_losses': [], 'G_losses': []}
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='x')
    z = tf.placeholder(tf.float32, shape=[None, 1, 1, latent], name='z')
    keep_prob = tf.placeholder(tf.float32)
    label = tf.placeholder(tf.float32, name='label', shape=[None, num_classes])
    labeled_mask = tf.placeholder(tf.float32, name='labeled_mask', shape=[None])
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

    lr_rate = 2e-4

    model = build_GAN(x, z, dropout_rate, is_training)
    D_real_features, D_real_logit, D_real_prob, D_fake_features, D_fake_logit, D_fake_prob, fake_data, D_real_mid_logit, D_fake_mid_logit= model
    extended_label = prepare_extended_label(label)

    # Fake_data of size = batch_size*28*28*1
    loss_acc = loss_accuracy(D_real_features, D_real_logit, D_real_prob,
                             D_fake_features, D_fake_logit, D_fake_prob,
                             extended_label, labeled_mask, D_real_mid_logit, D_fake_mid_logit)
    D_L, G_L, accuracy, prediction_value = loss_acc

    D_optimizer, G_optimizer = optimizer(D_L, G_L, lr_rate, beta1=0.5)
    saver = tf.train.Saver()  # defaults to saving all variables
    print('...开始训练...')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # mnist_data = get_data()
        # no_of_batches = int+(mnist_data.train.images.shape[0] / batch_size) + 1
        no_of_batches = int(data.train_X.shape[0] / batch_size) + 1
        print("训练样本数量:")
        print(data.train_X.shape[0])
        test_acc = []
        for epoch in range(epochs):

            train_accuracies, train_D_losses, train_G_losses = [], [], []

            for it in range(no_of_batches - 1):
                # batch = mnist_data.train.next_batch(batch_size, shuffle=False)
                # test_batch = mnist_data.train.next_batch(batch_size, shuffle=False)
                # batch[0] has shape: batch_size*28*28*1

                batch = data.train_X[it * batch_size:batch_size + it * batch_size, ]
                batch_label = data.train_Y[it * batch_size:batch_size + it * batch_size, ]
                # test_batch = mnist_data.train.next_batch(batch_size, shuffle=False)
                test_batch = data.test_X
                test_batch_label = data.test_Y
                Fina_test_batch = data.Fina_train_X
                Fina_batch_label = data.Fina_train_Y

                # batch_reshaped = tf.image.resize_images(batch[0], [64, 64]).eval()
                batch_reshaped = batch.reshape([-1, 32, 32, 1])
                test_batch_reshaped = test_batch.reshape([-1, 32, 32, 1])
                Fina_test_batch_reshaped = Fina_test_batch.reshape([-1, 32, 32, 1])

                # Reshaping the whole batch into batch_size*64*64*1 for disc/gen architecture
                batch_z = np.random.normal(0, 1, (batch_size, 1, 1, latent))

                mask = get_labeled_mask(labeled_rate, batch_size)

                train_feed_dict = {x: scale(batch_reshaped), z: batch_z,
                                   label: batch_label, labeled_mask: mask,
                                   dropout_rate: 0.7,
                                   is_training: True}

                test_feed_dict = {x: scale(test_batch_reshaped), z: batch_z,
                                  label: test_batch_label, labeled_mask: mask,
                                  dropout_rate: 0.7,
                                  is_training: True}
                # 字典中提供的是one—hot标签

                D_optimizer.run(feed_dict=train_feed_dict)
                G_optimizer.run(feed_dict=train_feed_dict)

                train_D_loss = D_L.eval(feed_dict=train_feed_dict)
                train_G_loss = G_L.eval(feed_dict=train_feed_dict)

                # Pre_value = prediction_value.eval(feed_dict=test_feed_dict)
                # print(Pre_value)
                train_accuracy = accuracy.eval(feed_dict=test_feed_dict)
                train_D_losses.append(train_D_loss)
                train_G_losses.append(train_G_loss)
                train_accuracies.append(train_accuracy)
                print('Epoch[{}/{}]'.format(epoch, epochs), 'Batch evaluated [{}]/[{}]'.format(str(it + 1), no_of_batches - 1))
            print('开始计算loss')
            tr_GL = np.mean(train_G_losses)
            tr_DL = np.mean(train_D_losses)
            tr_acc = np.mean(train_accuracies)
            print(train_accuracies)
            print(tr_acc)
            test_acc.append(tr_acc)
            print(test_acc)
            print('After epoch: ' + str(epoch + 1) + ' Generator loss: '
                  + str(tr_GL) + ' Discriminator loss: ' + str(tr_DL) + ' Accuracy: ' + str(tr_acc))

            gen_samples = fake_data.eval(
                feed_dict={z: np.random.normal(0, 1, (25, 1, 1, latent)), dropout_rate: 0.7, is_training: False})
            # Dont train batch-norm while plotting => is_training = False
            # test_images = tf.image.resize(gen_samples, [64, 64]).eval()
            # test_images = gen_samples
            # show_result(test_images, (epoch + 1), show=False, save=True)

            train_hist['D_losses'].append(np.mean(train_D_losses))
            train_hist['G_losses'].append(np.mean(train_G_losses))

        show_train_hist(train_hist, show=False, save=True)
        show_acc(test_acc, batch_size, epoch, show=False, save=True)
        print(len(test_acc))
        print(test_acc)
        acc_data = open("D:\data_acc.txt", 'w+')
        lenth = len(test_acc)

        for i in range(lenth):
            print(test_acc[i], file=acc_data)
        saver.save(sess, 'Gan_model/Ganmodel.ckpt')
        sess.close()
    return train_D_losses, train_G_losses


train_GAN(50, 100)

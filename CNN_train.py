import datetime
import os
import numpy as np
import tensorflow as tf
from openpyxl import load_workbook
from tensorflow import keras
import time
from ssgan_dataset_tf2 import test_Y, test_X, train_X, train_Y, valid_Y, valid_X
from CNN_model_tf2 import CNN
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


def prepare_extended_label(label):
    # add extra label for fake data
    extended_label = tf.concat([tf.zeros([tf.shape(label)[0], 1]), label], axis=1)

    return extended_label


def loss_fn(CNN_model, batch_x, batch_label, is_training):
    D_real_features, D_real_logits, D_real_prob = CNN_model(batch_x, is_training)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_label, logits=D_real_logits))
    return loss


def accuracy(Cnn, batch_x, extended_label, is_training):
    D_real_features, D_real_logits, D_real_prob = Cnn(batch_x, is_training)
    prediction_value = tf.argmax(D_real_prob[:, 1:], 1)
    prediction = tf.equal(tf.argmax(D_real_prob[:, 1:], 1),
                          tf.argmax(extended_label[:, 1:], 1))
    acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return acc, prediction_value


def Draw(hist, name, epoch, show=False, save=False, is_loss=True):
    plt.figure()
    Time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    if is_loss:
        plt.plot(hist, 'b', label='CNN')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            if not os.path.exists('Loss'):
                os.mkdir('Loss')
            plt.savefig("Loss/loss_lr[{}]epoch[{}]time[{}].png".format(name, epoch, Time))
    else:
        plt.plot(hist, 'b', label='acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        if save:
            if not os.path.exists('plot'):
                os.mkdir('plot')
            plt.savefig("plot/acc_lr[{}]epoch[{}]time[{}].png".format(name, epoch, Time))
    if show:
        plt.show()
    else:
        plt.close()


def main(learning_rate, epochs):
    batch_size = 64
    is_training = True
    Train_acc = []
    test_acc = []
    train_hist = []
    file = ['', '', '']  # 模型文件名称，删除之前保存的文件名称

    CNN_model = CNN()
    CNN_model.build(input_shape=(None, 32, 32, 1))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    CNN_model.summary()
    Time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = 'D:/python/ssgan_tf2.0/log_dir'
    log_dir = os.path.join(path, Time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = tf.summary.create_file_writer(log_dir)
    no_of_batches = int(train_X.shape[0] / batch_size) + 1
    for epoch in range(epochs):
        train_accuracies, train_losses = [], []
        for i in range(no_of_batches - 1):
            # 准备训练数据
            batch_x = train_X[i * batch_size:batch_size + i * batch_size, ]
            batch_label = train_Y[i * batch_size:batch_size + i * batch_size, ]
            batch_reshaped = batch_x.reshape([-1, 32, 32, 1])
            extended_label = prepare_extended_label(batch_label)

            # 准备验证数据
            valid_data = valid_X
            valid_label = valid_Y
            valid_data_reshaped = valid_data.reshape([-1, 32, 32, 1])
            valid_extended_label = prepare_extended_label(valid_label)

            # 判别器前向计算
            with tf.GradientTape() as tape:
                loss = loss_fn(CNN_model, batch_reshaped, extended_label, is_training)
                grads = tape.gradient(loss, CNN_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, CNN_model.trainable_variables))

            train_accuracy, _ = accuracy(CNN_model, valid_data_reshaped, valid_extended_label, None)
            train_accuracies.append(train_accuracy)
            print('Epoch [{}]/[{}]'.format(epoch, epochs),
                  'Batch evaluated [{}]/[{}]'.format(str(i + 1), no_of_batches - 1))

            train_losses.append(loss)
        tr_Loss = np.mean(train_losses)
        tr_acc = np.mean(train_accuracies)
        train_hist.append(tr_Loss)
        Train_acc.append(tr_acc)
        print('After epoch: ' + str(epoch + 1) + ' loss: ' + str(tr_Loss) + ' Accuracy: ' + str(tr_acc))

        with writer.as_default():
            tf.summary.scalar("train/tr_DL", tr_Loss, epoch)
            tf.summary.scalar("train/tr_acc", tr_acc, epoch)

        test_data = test_X
        test_label = test_Y
        test_data_reshaped = test_data.reshape([-1, 32, 32, 1])
        test_extended_label = prepare_extended_label(test_label)
        test_features, test_logits, test_prob = CNN_model(test_data_reshaped, False)
        test_accuracy, _ = accuracy(CNN_model, test_data_reshaped, test_extended_label, False)
        print('测试集：' + str(test_accuracy.numpy()))
        epoch_accuracy = test_accuracy
        with writer.as_default():
            tf.summary.scalar("test/test_accuracy", epoch_accuracy.numpy(), epoch)

        print(np.array(test_acc))
        print('目前准确率最大为' + str(tf.reduce_max(test_acc)))
        print(str(epoch_accuracy.numpy()), str(tf.reduce_max(test_acc).numpy()))
        if epoch_accuracy.numpy() >= tf.reduce_max(test_acc).numpy():
            Time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 本次 循环开始时间，放到文件命名
            for i in file:  # 删除上一个模型文件，保存新的模型
                if os.path.exists(i):
                    os.remove(i)
                else:
                    print('no such file:%s' % i)
            print('*************************模型保存***************************************')
            CNN_model.save_weights('CNN_model/model_time[{}]'.format(Time))
            file = ['CNN_model/model_time[{}].index'.format(Time),
                    'CNN_model/model_time[{}].data-00000-of-00002'.format(Time),
                    'CNN_model/model_time[{}].data-00001-of-00002'.format(Time)]
        test_acc.append(test_accuracy)

    return train_hist, Train_acc


if __name__ == '__main__':
    Learning_rate = 0.0005
    Epochs = 100
    train_loss, train_acc = main(Learning_rate, Epochs)
    Draw(train_loss, Learning_rate, Epochs, show=True, save=True)
    Draw(train_acc, Learning_rate, Epochs, show=True, save=True, is_loss=False)

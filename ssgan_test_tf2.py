import tensorflow as tf
from ssgan_train_tf2 import prepare_extended_label, accuracy
import ssgan_dataset_tf2
from tensorflow.keras import layers, Model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


class Discriminator(Model):
    # 判别器（分类器）
    def __init__(self, dropout_rate=0.5):
        super(Discriminator, self).__init__()
        filters = 64
        # 卷积层1
        self.conv1 = layers.Conv2D(filters, 4, 2, 'same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        # 卷积层2
        self.conv2 = layers.Conv2D(filters * 2, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        # 卷积层3
        self.conv3 = layers.Conv2D(filters * 4, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate)
        # 卷积层4
        self.conv4 = layers.Conv2D(filters * 8, 3, 1, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(dropout_rate)
        # 卷积层5
        self.conv5 = layers.Conv2D(filters * 16, 3, 1, 'same', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        # 全局池化层
        self.pool = layers.GlobalAveragePooling2D()
        # 特征打平层
        self.flatten = layers.Flatten()
        # 2 分类全连接层
        self.fc = layers.Dense(4)
        self.out_D = layers.Softmax()

    def call(self, inputs, training=None, reuse=False):
        # 卷积-BN-激活函数:(4, 31, 31, 64)
        x = self.dropout1(tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training)))
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x = self.dropout2(tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training)))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        x = self.dropout3(tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training)))
        # 卷积-BN-激活函数:(4, 4, 4, 512)
        x = self.dropout4(tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training)))
        # 卷积-BN-激活函数:(4, 2, 2, 1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        # 卷积-BN-激活函数:(4, 1024)
        x = self.pool(x)
        # 打平
        x_flatten = self.flatten(x)
        # 输出，[b, 1024] => [b, 4]
        x = self.fc(x_flatten)
        logits = self.out_D(x)

        return x_flatten, x, logits


discriminator = Discriminator()
discriminator.build(input_shape=(None, 64, 64, 1))

test_data = ssgan_dataset_tf2.test_X
test_label = ssgan_dataset_tf2.test_Y
test_data_reshaped = test_data.reshape([-1, 64, 64, 1])
test_extended_label = prepare_extended_label(test_label)

discriminator.load_weights('Gan_model/model_time[2020-07-14 12-33-46]')

test_accuracy, prediction_value = accuracy(discriminator, test_data_reshaped, test_extended_label, False)

print('准确率为：' + str(test_accuracy.numpy()))

# 打印预测标签到文件
prediction_value = prediction_value.numpy()
pre_dataw = open("result/data.txt", 'w+')
lenth1 = len(prediction_value)
for i in range(lenth1):
    print(prediction_value[i], file=pre_dataw)
pre_dataw.close()

# 打印真实标签到文件
real_value = tf.argmax(test_label, 1)
real_value = real_value.numpy()
real_dataw = open("result/realdata.txt", 'w+')
lenth2 = len(real_value)
for i in range(lenth2):
    print(real_value[i], file=real_dataw)
real_dataw.close()

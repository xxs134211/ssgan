
import tensorflow as tf
from tensorflow.keras import layers, Model


class Generator(Model):  # 生成器
    def __init__(self, dropout_rate=0.5):
        super(Generator, self).__init__()
        filters = 64
        # Layer 1   输入数据大小为[batch, 1, 1 ,100] 输出大小为512，核大小为4，步长为1，不使用padding
        self.conv1 = layers.Conv2DTranspose(filters * 8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)

        # Layer 2    输出大小为256，核大小为4，步长为4，使用padding
        self.conv2 = layers.Conv2DTranspose(filters * 4, 4, 4, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)

        # Layer 3    输出大小为128，核大小为4，步长为2，使用padding
        self.conv3 = layers.Conv2DTranspose(filters * 2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate)

        # Layer 4    64，核大小为4，步长为2，使用padding
        self.conv4 = layers.Conv2DTranspose(1, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None, reuse=False):
        x = inputs  # [z, 100]
        # Reshape 乘4D 张量，方便后续转置卷积运算:(b, 1, 1, 100)
        x = tf.reshape(x, (x.shape[0], 1, 1, 100))
        x = tf.nn.relu(x)  # 激活函数
        # 转置卷积-BN-激活函数-dropout:(b, 4, 4, 512)
        x = self.dropout1(tf.nn.relu(self.bn1(self.conv1(x), training=training)))
        # 转置卷积-BN-激活函数:(b, 16, 16, 256)
        x = self.dropout2(tf.nn.relu(self.bn2(self.conv2(x), training=training)))
        # 转置卷积-BN-激活函数:(b, 32, 32, 128)
        x = self.dropout3(tf.nn.relu(self.bn3(self.conv3(x), training=training)))
        # 转置卷积-BN-激活函数:(b, 64, 64, 1)
        x = self.conv4(x)
        x = tf.image.resize(x, [32, 32])  # ？*32*32*1
        x = tf.tanh(x)  # 输出x 范围-1~1,与预处理一致

        return x


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
        x1 = self.dropout1(tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training)))
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x2 = self.dropout2(tf.nn.leaky_relu(self.bn2(self.conv2(x1), training=training)))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        x3 = self.dropout3(tf.nn.leaky_relu(self.bn3(self.conv3(x2), training=training)))
        # 卷积-BN-激活函数:(4, 4, 4, 512)
        x4 = self.dropout4(tf.nn.leaky_relu(self.bn4(self.conv4(x3), training=training)))
        # 卷积-BN-激活函数:(4, 2, 2, 1024)
        x5 = tf.nn.leaky_relu(self.bn5(self.conv5(x4), training=training))
        # 卷积-BN-激活函数:(4, 1024)
        x = self.pool(x5)
        # 打平
        x_flatten = self.flatten(x)
        # 输出，[b, 1024] => [b, 4]
        x = self.fc(x_flatten)
        logits = self.out_D(x)

        return x_flatten, x, logits, x3

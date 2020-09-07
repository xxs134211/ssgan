class Discriminator(Model):
    # 判别器（分类器）
    def __init__(self, dropout_rate=0.5):
        super(Discriminator, self).__init__()
        filters = 32
        # 卷积层1
        self.conv1 = layers.Conv2D(filters, 4, 2, 'same', use_bias=False)
        self.bn1 = layers.MaxPool2D(pool_size=2, strides=2)
        self.dropout1 = layers.Dropout(dropout_rate)
        # 卷积层2
        self.conv2 = layers.Conv2D(filters * 2, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.MaxPool2D(pool_size=2, strides=2)
        self.dropout2 = layers.Dropout(dropout_rate)
        # 特征打平层
        # self.flat = tf.reshape(shape=[-1, 7 * 7 * 64])
        self.flatten = layers.Flatten()

        # 2 分类全连接层
        self.fc = layers.Dense(4)
        self.out_D = layers.Softmax()

    def call(self, inputs, training=None, reuse=False):
        # 卷积-BN-激活函数:(4, 31, 31, 64)
        x1 = self.dropout1(tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training)))
        # print(x1.shape)
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x2 = self.dropout2(tf.nn.leaky_relu(self.bn2(self.conv2(x1), training=training)))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        # print(x2.shape)
        # 卷积-BN-激活函数:(4, 1024)
        x = tf.reshape(x2, [-1, 2 * 2 * 64])
        # 打平
        x_flatten = self.flatten(x)
        # 输出，[b, 1024] => [b, 4]
        x = self.fc(x_flatten)
        logits = self.out_D(x)

        return x_flatten, x, logits, x2
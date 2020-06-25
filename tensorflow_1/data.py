from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

import warnings

warnings.filterwarnings("ignore")


def prepro(d_path, length=1024, number=1000, normal=True, rate=None, enc=True, enc_step=28):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名
    if rate is None:
        rate = [0.5, 0.25, 0.25]
    filenames = os.listdir(d_path)  # 这里filenames为一个list

    def capture(original_path):

        """读取mat文件，返回字典
        :param original_path: 读取路径
        :return: 数据字典
        """

        files = {}  # 创建一个空字典

        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)  # 获取到d_path目录下所有文件地址
            file = loadmat(file_path)  # 加载.mat文件中所有的结构体
            file_keys = file.keys()  # 返回字典file中所有的键
            for key in file_keys:  # 获取所有文件中结构体中含有字符为DE的数据，并将数据写入字典中
                if 'GW' in key:
                    files[i] = file[key].ravel()
                    print("file: ", files[i])
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):

        """将数据切分为前面多少比例，后面多少比例.
        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()  # 获取字典数据data中的键
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]  # 获取字典data中所有的值，即振动数据
            all_lenght = len(slice_data)  # 获取各个数据的长度
            end_index = int(all_lenght * (1 - slice_rate))  # 训练集数据个数
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enc:  # 采用数据增强(True)
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数

                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0

                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break

                    if label:
                        break

            else:
                for j in range(samp_train):  # 样本总数
                    random_start = np.random.randint(low=0, high=(end_index - length))  # 生成0到(end_index - length)的随机数
                    sample = slice_data[random_start:random_start + length]  # 从生成的随机数开始，往后length个点作为一段测试数据
                    Train_sample.append(sample)

            # 抓取测试数据

            for h in range(number - samp_train):  # 总样本数-训练样本数
                random_start = np.random.randint(low=end_index,
                                                 high=(all_lenght - length))  # 生成end_index到(all_length-length)的随机数
                sample = slice_data[random_start:random_start + length]  # 从生成的随机数开始，往后length个点作为一段测试数据
                Test_Sample.append(sample)  # 将测试样本放入list当中

            Train_Samples[i] = Train_sample  # 将列表数据Train_sample放入字典Train_Samples[i]中
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples  # 该字典的键名为文件名12k_Drive_End_B007_0_118.mat等

    # 仅抽样完成，打标签

    def add_labels(train_test):  # train_test 为字典
        X = []
        Y = []
        label = 0

        for i in filenames:
            # print("字典", i, ": ", train_test[i])
            # print(i, ": ", label)
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx  # 生成一个长度为lenx(这里使500)的list,每个值都为label
            label += 1

        # print("X: ", X.shape)
        # print("Y: ", len(Y))
        return X, Y  # X为一个长度为2000的list,其中每个数据为2048个点的振动数据；Y为对应的标签，也是一个长度为2000的list,分别为0，1，2，3,各500个,与X对应。

    # one-hot编码

    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])  # 把训练集标签转换成n行一列
        Test_Y = np.array(Test_Y).reshape([-1, 1])  # 把测试集标签转换成n行一列
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(
            Train_Y).toarray()  # 将训练集标签转换为onehot编码，即由原来的0，1，2，3转换为数组[[1.0.0.0.],[0.1.0.0.],[0.0.1.0.],[0.0.0.1.]]
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)  # 转换成整型?
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])  # 0.5
        ss = StratifiedShuffleSplit(n_splits=1,
                                    test_size=test_size)  # 这里n_splits表示数据划分的组数，每组数据是相同的，只是顺序不同；test_size表示测试集在一组数据中所占的比例。
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]

            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # print('DATA: ', data)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # print("train: ", len(train))
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # print(Train_X.shape)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        # print("normal??")
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        # print("dddddmm")
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    # 打乱数据，保证测试均匀
    state = np.random.get_state()
    np.random.shuffle(Train_X)
    # print(train_X)
    np.random.set_state(state)
    np.random.shuffle(Train_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


# if __name__ == "__main__":
# path = r'D:\pycharm\worksapce\Resnet\1DCNN\data\0HP'
path = 'D:/train'
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                            length=1024,
                                                            number=500,
                                                            normal=False,
                                                            rate=[0.8, 0.1, 0.1],
                                                            enc=False,
                                                            enc_step=20)

Fina_train_X, Fina_train_Y, Fina_valid_X, Fina_valid_Y, Fina_test_X, Fina_test_Y = prepro(d_path='D:/n_test',
                                                                                          length=1024,
                                                                                          number=200,
                                                                                          normal=False,
                                                                                          rate=[0.5, 0.25, 0.25],
                                                                                          enc=False,
                                                                                          enc_step=20)

print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)
print(Fina_train_X.shape)
print("**********数据处理完毕***************")
print("**********开始建立模型***************")

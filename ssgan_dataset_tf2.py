from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import warnings

warnings.filterwarnings("ignore")


def DataSet(d_path, length=1024, number=1000, normal=True, rate=None, enc=True, enc_step=28):
    """
    对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本
    :param d_path: 样本数据地址
    :param length: 单个样本长度
    :param number: 每一类数据的个数，默认每一类数据1000个数据
    :param normal: 是否标准化
    :param rate: 训练集、验证集、测试集的比例，默认[0.5,0.25,0.25]，相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    """

    if rate is None:
        rate = [0.5, 0.25, 0.25]
    filenames = os.listdir(d_path)

    def getdata():
        """
        读取ma他文件，返回字典
        :return: 数据字典
        """
        files = {}
        get_data = {}
        for i in filenames:
            file_path = os.path.join(d_path, i)  # 获取到d_path目录下所有文件地址
            file = loadmat(file_path)  # 加载.mat文件中所有的结构体
            file_keys = file.keys()
            for key in file_keys:  # 获取所有文件中结构体中含有字符为DE的数据，并将数据写入字典中
                if 'M07' in key:
                    files[i] = file[key]
                    get_data[i] = files[i][0, 0]['Y'][0, 6][2].ravel()

                if 'data_channel_2' in key:
                    files[i] = file[key]
                    get_data[i] = files[i].ravel()

        return get_data

    def slice_enc(slicedata, slice_rate=rate[1] + rate[2]):
        """
        将数据切分为前面多少比例，后面多少比例.
        :param slicedata: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        # print(keys)
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = slicedata[i]  # 获取字典data中所有的值，即振动数据
            all_lenght = len(slice_data)  # 获取样本的长度
            end_index = int(all_lenght * (1 - slice_rate))  #
            samp_train = int(number * (1 - slice_rate))  # 500  训练集数据个数
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
                # 生成end_index到(all_length-length)的随机数
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]  # 从生成的随机数开始，往后length个点作为一段测试数据
                Test_Sample.append(sample)  # 将测试样本放入list当中

            Train_Samples[i] = Train_sample  # 将列表数据Train_sample放入字典Train_Samples[i]中
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签

    def add_labels(train_test):  # train_test 为字典
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx  # 生成一个长度为lenx的list,每个值都为label
            label += 1
        return X, Y  # 已经将所有类别样本放在一起了，不再是按字典存放了

    # one-hot编码

    def one_hot(Train_labelY, Test_labelY):
        Train_labelY = np.array(Train_labelY).reshape([-1, 1])  # 把训练集标签转换成n行一列
        Test_labelY = np.array(Test_labelY).reshape([-1, 1])  # 把测试集标签转换成n行一列
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_labelY)
        Train_labelY = Encoder.transform(
            Train_labelY).toarray()  # 将训练集标签转换为onehot编码，即由原来的0，1，2，3转换为数组[[1.0.0.0.],[0.1.0.0.],[0.0.1.0.],[0.0.0.1.]]
        Test_labelY = Encoder.transform(Test_labelY).toarray()
        Train_labelY = np.asarray(Train_labelY, dtype=np.int32)  # 转换成整型?
        Test_labelY = np.asarray(Test_labelY, dtype=np.int32)
        return Train_labelY, Test_labelY

    def scalar_stand(Train_dataX, Test_dataX):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_dataX)
        Train_dataX = scalar.transform(Train_dataX)
        Test_dataX = scalar.transform(Test_dataX)
        return Train_dataX, Test_dataX

    def valid_test_slice(Test_dataX, Test_labelY):
        test_size = rate[2] / (rate[1] + rate[2])  # 0.5
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        # 这里n_splits表示数据划分的组数，每组数据是相同的，只是顺序不同；test_size表示测试集在一组数据中所占的比例。
        for train_index, test_index in ss.split(Test_dataX, Test_labelY):
            X_valid, X_test = Test_dataX[train_index], Test_dataX[test_index]
            Y_valid, Y_test = Test_labelY[train_index], Test_labelY[test_index]

            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = getdata()
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)

    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)

    # 打乱数据，保证测试均匀
    state = np.random.get_state()
    np.random.shuffle(Train_X)
    np.random.set_state(state)
    np.random.shuffle(Train_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


path_train = 'D:/python/bearing_data/train1'
# path_train = 'D:/python/bearing_data/北交辛格数据/train_data'
# path_train = 'D:/python/bearing_data/北交赵雪军数据/test'
train_X1, train_Y1, valid_X1, valid_Y1, test_X1, test_Y1 = DataSet(d_path=path_train,
                                                                   length=4096,
                                                                   number=1000,
                                                                   normal=False,
                                                                   rate=[0.7, 0.2, 0.1],
                                                                   enc=True,
                                                                   enc_step=50)
# print(train_X1, train_Y1)
# print(test_X1.shape, test_Y1.shape)
# path_test = 'D:/python/DANN/dataset/data_train'
path_test = 'D:/python/bearing_data/test1'
# path_test = 'D:/python/bearing_data/北交辛格数据/test_data'
# path_test = 'D:/python/bearing_data/北交赵雪军数据/ball'
train_X2, train_Y2, valid_X2, valid_Y2, test_X2, test_Y2 = DataSet(d_path=path_test,
                                                                   length=4096,
                                                                   number=600,
                                                                   normal=False,
                                                                   rate=[0.1, 0.1, 0.8],
                                                                   enc=False,
                                                                   enc_step=50)

train_X, train_Y, valid_X, valid_Y, test_X, test_Y = train_X1, train_Y1, valid_X1, valid_Y1, test_X2, test_Y2
print("训练样本=" + str(train_X.shape[0]))
print("验证样本=" + str(valid_X.shape[0]))
print("测试样本=" + str(test_X.shape[0]))
print("**********数据处理完毕***************")
print("**********开始建立模型***************")
# print(tf.argmax(test_Y, 1))

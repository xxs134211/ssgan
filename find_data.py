import numpy as np
import xlrd
import xlwt


def per_data(tag):
    # 打开文件
    workBook = xlrd.open_workbook('D:/python/data.xlsx')
    # 2.1 法1：按索引号获取sheet内容
    sheet1_content1 = workBook.sheet_by_index(0)  # sheet索引从0开始
    sheet2_content1 = workBook.sheet_by_index(1)
    index_model_1 = []
    index_model_2 = []
    for i in range(10):
        cols = sheet1_content1.col_values(i)
        for j in range(100):
            maxdata = max(cols[1:j + 2])
            if maxdata >= tag:
                index_model_1.append(j + 1)
                break
    index_model_1 = np.mean(index_model_1)
    for i in range(10):
        cols = sheet2_content1.col_values(i)
        for j in range(100):
            maxdata = max(cols[1:j + 2])
            if maxdata >= tag:
                index_model_2.append(j + 1)
                break
    index_model_2 = np.mean(index_model_2)
    print('模型一达到{}准确率的平均次数为{}'.format(tag, index_model_1))
    print('模型二达到{}准确率的平均次数为{}'.format(tag, index_model_2))


# for i in range(80, 98):
#     per_data(i/100)


workBook = xlrd.open_workbook('D:/python/data.xlsx')
# 2.1 法1：按索引号获取sheet内容
sheet1_content1 = workBook.sheet_by_index(1)  # sheet索引从0开始
# rows = sheet1_content1.row_values(100)
a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in a:
    rows = sheet1_content1.row_values(i)
    acc_avg = np.mean(rows)
    print('模型一循环{}次之后的平均准确率为:{}'.format(i, acc_avg))


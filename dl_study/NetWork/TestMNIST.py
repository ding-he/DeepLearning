import struct
import NetWork as nw
from datetime import datetime

class Loader:
    '''
    数据加载器基类
    '''
    def __init__(self, path, count):
        '''
        初始化加载器
        path:       数据文件路径
        count:      文件中的样本个数
        '''
        self.path = path
        self.count = count

    def getFileContent(self):
        '''
        读取文件内容
        '''
        with open(self.path, 'rb') as f:
            content = f.read()
        
        return list(content)


class ImageLoader(Loader):
    '''
    图像数据加载器
    '''

    def getPicture(self, content, index):
        '''
        从文件中读取图像
        '''
        # 数据开始索引
        start = index * 28 * 28 + 16

        # 一幅图像
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    content[start + i*28 + j]
                )

        return picture

    def getSample(self, picture):
        '''
        将图像转换为样本输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        
        return sample

    def load(self):
        '''
        加载数据文件
        获得全部样本的输入向量
        '''
        content = self.getFileContent()

        # 归一化
        for i in range(len(content)):
            content[i] /= 255

        dataSet = []
        for index in range(self.count):
            dataSet.append(self.getSample(self.getPicture(content, index)))

        return dataSet


class LabelLoader(Loader):
    '''
    标签数据加载器
    '''
    def load(self):
        '''
        加载数据文件
        获得全部样本的标签向量
        '''
        content = self.getFileContent()
        
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))

        return labels

    def norm(self, label):
        '''
        将标签值转换为10维标签向量
        '''
        labelVector = []
        labelValue = label

        for i in range(10):
            if i == labelValue:
                labelVector.append(0.9)
            else:
                labelVector.append(0.1)

        return labelVector


def getTrainDataSet():
    '''
    获取训练数据集
    '''
    imageLoader = ImageLoader(
        'E:/Documents/Test/test_data/MNIST/train-images.idx3-ubyte',
        60000
    )
    labelLoader = LabelLoader(
        'E:/Documents/Test/test_data/MNIST/train-labels.idx1-ubyte',
        60000
    )

    return imageLoader.load(), labelLoader.load()


def getTestDataSet():
    '''
    获取测试数据集
    '''
    imageLoader = ImageLoader(
        'E:/Documents/Test/test_data/MNIST/t10k-images.idx3-ubyte',
        10000
    )
    labelLoader = LabelLoader(
        'E:/Documents/Test/test_data/MNIST/t10k-labels.idx1-ubyte',
        10000
    )

    return imageLoader.load(), labelLoader.load()


def getResult(vector):
    '''
    获取输出结果
    '''
    maxValueIndex = 0
    maxValue = vector[0]

    # 找出最大结果输出
    for i in range(1, len(vector)):
        if vector[i] > maxValue:
            maxValue = vector[i]
            maxValueIndex = i

    return maxValueIndex


def evaluate(network, dataSet, labels):
    '''
    使用错误率评估网络
    '''
    errorCount = 0
    totalCount = len(dataSet)

    for i in range(totalCount):
        # 实际值
        label = getResult(labels[i])
        
        # 预测值
        predict = getResult(network.predict(dataSet[i]))

        if label != predict:
            errorCount += 1
    
    # 返回错误率
    return float(errorCount) / float(totalCount)


def trainEvaluate():
    '''
    训练10轮, 进行一次评估
    '''
    lastErrorRatio = 1.0
    epoch = 0

    # 获取训练和测试数据集
    trainDataSet, trainLabels = getTrainDataSet()
    testDataSet, testLabels = getTestDataSet()

    print('get data set finished...')

    # 构建全连接网络
    network = nw.Network([784, 300, 10])

    while True:
        epoch += 1

        # 一次训练
        network.train(trainLabels, trainDataSet, 0.3, 1)
        print('%s epoch %d finished' % (datetime.now(), epoch))

        if epoch % 10 == 0:
            # 进行一次评估
            errorRatio = evaluate(network, testDataSet, testLabels)
            print(
                '%s after epoch %d, error ratio is %f' % 
                datetime.now(), epoch, errorRatio
            )

            if errorRatio > lastErrorRatio:
                break
            else:
                lastErrorRatio = errorRatio
            

if __name__ == '__main__':
    trainEvaluate()

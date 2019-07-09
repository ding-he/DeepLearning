import numpy as np

class FullConnectedLayer:
    '''
    全连接层
    '''

    def __init__(self, inputSize, outputSize, activator):
        '''
        构造全连接网络
        inputSize:      本层输入向量的维度
        outputSize:     本层输出向量的维度
        '''
        self.inputSize = inputSize
        self.outputSize = outputSize

        # 激活函数
        self.activator = activator

        # 权重矩阵
        self.weight = np.random.uniform(-0.1, 0.1, (outputSize, inputSize))

        # 偏置项
        self.bias = np.zeros((outputSize, 1))

        # 输出向量
        self.output = np.zeros((outputSize, 1))

    def forward(self, inputArray):
        '''
        前向计算, 计算本层的输出值
        inputArray:     输入向量, 维度 = inputSize
        '''
        self.input = inputArray

        # 向量相乘得到输出
        self.output = self.activator.forward(
            np.dot(self.weight, inputArray) + self.bias
        )

    def backward(self, deltaArray):
        '''
        反向计算weight和b的梯度
        deltaArray:     上一层传来的误差项
        '''
        # 本层的delta
        self.delta = self.activator.backward(self.input) \ 
            * np.dot(np.transpose(self.weight), deltaArray)
        self.weightGrad = np.dot(deltaArray, np.transpose(self.input))
        self.biasGrad = deltaArray

    def update(self, rate):
        '''
        梯度下降算法更新权重
        '''
        self.weight += rate * self.weightGrad
        self.bias += rate * self.biasGrad


class SigmoidActivator:
    '''
    sigmoid激活函数类
    '''
    def forward(self, weightedInput):
        '''
        前向计算输出值
        '''
        return 1.0 / (1.0 + np.exp(-weightedInput))

    def backward(self, output):
        '''
        反向计算
        '''
        return output * (1 - output)


class Network:
    '''
    神经网络类
    '''
    def __init__(self, layers):
        '''
        构造神经网络
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i + 1],
                    SigmoidActivator()
                )
            )
    
    def predict(self, sample):
        '''
        预测输出
        sample:     样本输入
        '''
        input = sample
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
        
        return input

    def train(self, labels, dataSet, rate, epoch):
        '''
        训练样本
        labels:     样本标签
        dataSet:    输入样本
        rate:       学习速率
        epoch:      训练轮数
        '''
        for i in range(epoch):
            for j in range(len(dataSet)):
                self.trainSample(
                    labels[j].reshape((labels[j].shape[0], 1)),
                    dataSet[j].reshape((dataSet[j].shape[0], 1)),
                    rate
                )

    def trainSample(self, label, sample, rate):
        '''
        训练一次样本
        '''
        self.predict(sample)
        self.calculateGradient(label)
        self.updateWeight(rate)

    def calculateGradient(self, label):
        '''
        计算梯度
        '''
        # 先计算最后一层
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)

        # 依次从后往前计算
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        
        return delta

    def updateWeight(self, rate):
        '''
        更新权重
        '''
        for layer in self.layers:
            layer.update(rate)

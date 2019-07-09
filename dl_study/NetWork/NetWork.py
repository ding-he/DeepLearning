from functools import reduce
import random
import numpy as np


class Node:
    '''
    节点类
    记录该节点的自身信息以及上下游连接
    实现输出值和误差项的计算
    '''

    def __init__(self, layerIndex, nodeIndex):
        '''
        构造节点对象
        layerIndex:     节点所属层的编号
        nodeIndex:      节点的编号
        '''
        self.layerIndex = layerIndex
        self.nodeIndex = nodeIndex

        # 上下游连接
        self.downStream = []
        self.upStream = []

        # 输出值和误差值
        self.output = 0.0
        self.delta = 0.0

    def setOutput(self, data):
        '''
        设置输入层节点的输出
        '''
        self.output = data

    def appendDownStreamConnection(self, connection):
        '''
        添加一个下游节点
        '''
        self.downStream.append(connection)

    def appendUpStreamConnection(self, connection):
        '''
        添加一个上游节点
        '''
        self.upStream.append(connection)

    def calculateOutput(self):
        '''
        计算节点的输出
        '''
        output = reduce(
            lambda value, conn: value + conn.upStreamNode.output \ 
                * conn.weight,
            self.upStream, 0.0
        )
        self.output = sigmoid(output)

    def calculateHiddenLayerDelta(self):
        '''
        计算隐藏层节点的delta
        '''

        # 先计算下游节点的delta与权重的值
        downDelta = reduce(
            lambda value, conn: value + conn.downStreamNode.delta \
                * conn.weight,
            self.downStream, 0.0
        )

        # 计算隐藏层的delta
        self.delta = self.output * (1 - self.output) * downDelta

    def calculateOutputLayerDelta(self, label):
        '''
        计算输出层节点的delta
        '''
        self.delta = self.output * (1 - self.output) \
            * (label - self.output)

    def __str__(self):
        '''
        打印节点信息
        '''
        nodeStr = '%u - %u: output: %f delta: %f' % (
            self.layerIndex, self.nodeIndex, self.output, self.delta)
        downStr = reduce(lambda value, conn: value + '\n\t' +
                         str(conn), self.downStream, '')
        upStr = reduce(lambda value, conn: value + '\n\t' +
                       str(conn), self.upStream, '')
        return nodeStr + '\n\tdown stream:' + downStr \
            + '\n\tup stream:' + upStr


class ConstNode:
    '''
    输出为1的节点
    '''

    def __init__(self, layerIndex, nodeIndex):
        '''
        构造节点对象
        layerIndex:     节点所属层的编号
        nodeIndex:      节点的编号
        '''
        self.layerIndex = layerIndex
        self.nodeIndex = nodeIndex

        # 下游连接
        self.downStream = []

        # 数尺
        self.output = 1.0

    def appendDownStreamConnection(self, connection):
        '''
        添加连接到下游节点中
        '''
        self.downStream.append(connection)

    def calculateHiddenLayerDelta(self):
        '''
        计算隐藏层节点的delta
        '''
        downDelta = reduce(
            lambda value, conn: value + conn.downStreamNode.delta \
                * conn.weight,
            self.downStream, 0.0
        )
        self.delta = self.output * (1 - self.output) * downDelta

    def __str__(self):
        '''
        打印节点信息
        '''
        nodeStr = '%u - %u: output: %f delta: %f' % (
            self.layerIndex, self.nodeIndex, self.output, self.delta)
        downStr = reduce(lambda value, conn: value + '\n\t' +
                         str(conn), self.downStream, '')
        upStr = reduce(lambda value, conn: value + '\n\t' +
                       str(conn), self.upStream, '')
        return nodeStr + '\n\tdown stream:' + downStr \
            + '\n\tup stream:' + upStr


class Layer:
    '''
    网络中的层
    '''

    def __init__(self, layerIndex, nodeCount):
        '''
        初始化层
        layerIndex:     层序号
        nodeCount:      该层的节点数
        '''
        self.layerIndex = layerIndex
        self.nodes = []

        # 初始化所有节点
        for i in range(nodeCount):
            self.nodes.append(Node(layerIndex, i))
        
        # 添加一个常量节点
        self.nodes.append(ConstNode(layerIndex, nodeCount))

    def setOutput(self, data):
        '''
        设置输入层的输出
        '''
        for i in range(len(data)):
            self.nodes[i].setOutput(data[i])

    def calculateOutput(self):
        '''
        计算层的输出向量
        '''
        for node in self.nodes[:-1]:
            node.calculateOutput()
    
    def dump(self):
        '''
        打印层的信息
        '''
        for node in self.nodes:
            print(node)


class Connection:
    '''
    连接类
    记录连接的权重
    记录上下游节点
    '''

    def __init__(self, upStreamNode, downStreamNode):
        '''
        初始化连接
        upStreamNode:   上游节点
        downStreamNode: 下游节点
        '''
        self.upStreamNode = upStreamNode
        self.downStreamNode = downStreamNode

        # 初始权重是一个小的随机数
        self.weight = random.uniform(-0.1, 0.1)

        # 初始化梯度值
        self.gradient = 0.0

    def calculateGradient(self):
        '''
        计算梯度
        梯度 = 下游节点的delta * 上游节点的输出
        '''
        self.gradient = self.downStreamNode.delta \
            * self.upStreamNode.output

    def getGradient(self):
        '''
        获取当前梯度值
        '''
        return self.gradient

    def updateWeight(self, rate):
        '''
        根据梯度下降算法更新权重
        '''
        self.calculateGradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u - %u) -> (%u - %u) = %f' % (
            self.upStreamNode.layerIndex,
            self.upStreamNode.nodeIndex,
            self.downStreamNode.layerIndex,
            self.upStreamNode.nodeIndex,
            self.weight
        )


class Connections:
    '''
    对Connection提供集合操作
    '''
    
    def __init__(self):
        '''
        初始化Connection集合
        '''
        self.connections = []

    def addConnection(self, connection):
        '''
        添加一个连接到集合中
        '''
        self.connections.append(connection)

    def dump(self):
        '''
        打印连接集合信息
        '''
        for connection in self.connections:
            print(connection)


class Network:
    '''
    全连接神经网络
    '''

    def __init__(self, layers):
        '''
        初始化全连接神经网络
        layers:     二维数组, 描述神经网络每层的节点数
        '''
        # 连接层集合
        self.connections = Connections()
        
        self.layers = []
        layerCount = len(layers)
        nodeCount = 0

        # 添加层
        for i in range(layerCount):
            self.layers.append(Layer(i, layers[i]))

        # 构建连接集合
        for layer in range(layerCount - 1):
            connections = [
                Connection(upStreamNode, downStreamNode)
                for upStreamNode in self.layers[layer].nodes
                for downStreamNode in self.layers[layer + 1].nodes[:-1]
            ]

            for connection in connections:
                self.connections.addConnection(connection)

                # 对节点添加上下游连接
                connection.downStreamNode.appendUpStreamConnection(
                    connection
                )
                connection.upStreamNode.appendDownStreamConnection(
                    connection
                )                

    def train(self, labels, dataSet, rate, interation):
        '''
        训练神经网络
        labels:     训练样本的标签
        dataSet:    二维数据集
        '''
        for i in range(interation):
            for j in range(len(dataSet)):
                self.trainSample(labels[j], dataSet[j], rate)

    def trainSample(self, label, sample, rate):
        '''
        训练单个样本集
        '''
        self.predict(sample)
        self.calculateDelta(label)
        self.updateWeight(rate)

    def calculateDelta(self, label):
        '''
        计算每个节点的delta
        '''
        # 输出层的系节点集合
        outputNodes = self.layers[-1].nodes

        # 计算输出层节点的delta
        for i in range(len(label)):
            outputNodes[i].calculateOutputLayerDelta(label[i])
        
        # 计算隐藏层节点的delta
        # 从倒数第二层开始取数
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calculateHiddenLayerDelta();

    def updateWeight(self, rate):
        '''
        更新每个连接的权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connection in node.downStream:
                    connection.updateWeight(rate)

    def calculateGradient(self):
        '''
        计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connection in node.downStream:
                    connection.calculateGradient()

    def getGradient(self, sample, label):
        '''
        获取在一个样本下, 每个连接的梯度
        sample:     样本输入
        label:      样本标签
        '''
        self.predict(sample)
        self.calculateDelta(label)
        self.calculateGradient()

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sample:     样本的输入向量
        '''
        
        # 输入层的输出
        self.layers[0].setOutput(sample)

        # 其它层的输出
        for i in range(1, len(self.layers)):
            self.layers[i].calculateOutput()
        
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        '''
        打印神经网络信息
        '''
        for layer in self.layers:
            layer.dump()


def checkGradient(network, sampleFeature, sampleLabel):
    '''
    梯度检查
    network:        神经网络对象
    sampleFeature:  样本的特征
    sampleLabel:    样本的标签
    '''
    # 计算网络的误差
    # 误差函数
    networkError = lambda vec1, vec2: \
        0.5 * reduce(
            lambda a, b: a + b,
            map(
                lambda v: (v[0] - v[1]) ** 2,
                zip(vec1, vec2)
            )
        )
    
    # 获取神经网络在当前样本下的每个连接的梯度
    network.getGradient(sampleFeature, sampleLabel)

    # 对每个权重做梯度检查
    for connection in network.connections.connections:
        # 获取指定连接的梯度
        actualGradient = connection.getGradient()

        # 增加一个很小的值, 计算网络的误差
        epsilon = 0.0001
        connection.weight += epsilon
        error1 = networkError(network.predivt(sampleFeature), sampleLabel)

        # 减去一个很小的值, 计算网络的误差
        connection.weight -= 2 * epsilon
        error2 = networkError(network.predivt(sampleFeature), sampleLabel)

        # 计算期望的梯度值
        expectedGradient = (error2 - error1) / (2 * epsilon)

        # 打印结果
        print(
            'expected gradient:\t%f\nactual gradient:\t%f' % 
            expectedGradient, actualGradient
        )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

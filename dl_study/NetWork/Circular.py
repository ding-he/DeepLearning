import numpy as np
from functools import reduce
from Conv import ReluActivator, elementWise

class CircularLayer:
    '''
    循环卷积层
    '''
    def __init__(
        self,
        inputWidth,
        stateWidth,
        activator,
        learingRate):
        '''
        构造循环卷积网络
        inputWidth:     输入层宽度
        stateWidth:     状态传递宽度
        activator:      激活函数
        learingRate:    学习率
        '''
        self.inputWidth = inputWidth
        self.stateWidth = stateWidth
        self.activator = activator
        self.learingRate = learingRate

        # 初始化当前时刻为0
        self.times = 0

        # 保存各个时刻的状态输出
        self.stateList = []

        # 初始化s0
        self.stateList.append(np.zeros((stateWidth, 1)))

        # 初始化U
        self.U = np.random.uniform(-1e-4, 1e-4, (stateWidth, inputWidth))

        # 初始化W
        self.W = np.random.uniform(-1e-4, 1e-4, (stateWidth, stateWidth))

    def forward(self, inputArray):
        '''
        前向计算输出值
        '''
        self.times += 1
        state = np.dot(self.U, inputArray) + \
            np.dot(self.W, self.stateList[-1])
        
        elementWise(state, self.activator.forward)
        self.stateList.append(state)

    def backward(self, sensitivityArray, activator):
        '''
        反向计算误差项
        '''
        self.calculateDelta(sensitivityArray, activator)
        self.calculateGradient()

    def calculateDelta(self, sensitivityArray, activator):
        '''
        计算delta
        '''
        # 保存每个时刻的误差项
        self.deltaList = []

        for i in range(self.times):
            # 初始化为0
            self.deltaList.append(np.zeros((self.stateWidth, 1)))
        self.deltaList.append(sensitivityArray)

        # 迭代计算每个时刻的误差项
        for i in range(self.times - 1, 0, -1):
            self.calculateDeltaSingle(i, activator)
    
    def calculateDeltaSingle(self, time, activator):
        '''
        计算单次的误差项
        根据time + 1时刻的误差项计算time时刻的误差项
        '''
        state = self.stateList[time + 1].copy()
        elementWise(self.stateList[time + 1], activator.backward)

        self.deltaList[time] = np.transpose(np.dot(
            np.dot(np.transpose(self.deltaList[time + 1]), self.W),
            np.diag(state[:, 0])
        ))

    def calculateGradient(self):
        '''
        计算梯度值
        '''
        # 保存每个时刻的权重梯度
        self.gradientList = []
        for i in range(self.times + 1):
            self.gradientList.append(np.zeros(
                (self.stateWidth, self.stateWidth)
            ))
        
        for i in range(self.times, 0, -1):
            self.calculateGradientSingle(i)

        # 计算各个时刻梯度之和
        self.gradient = reduce(
            lambda a, b: a + b,
            self.gradientList,
            self.gradientList[0]
        )

    def calculateGradientSingle(self, time):
        '''
        计算单个时刻的梯度
        '''
        gradient = np.dot(
            self.deltaList[time],
            np.transpose(self.stateList[time - 1])
        )

        self.gradientList[time] = gradient

    def update(self):
        '''
        使用梯度下降算法更新权重
        '''
        self.W -= self.learingRate * self.gradient

    def resetState(self):
        '''
        初始化状态
        '''
        self.times = 0
        self.stateList = []
        self.stateList.append(np.zeros(
            (self.stateWidth, 1)
        ))


def gradientCheck():
    '''
    梯度检查
    '''
    # 误差函数, 取所有节点输出项之和
    errorFunction = lambda o: o.sum()

    cl = CircularLayer()

    # 计算forward输出值
    x, d = getDataSet()
    cl.forward(x[0])
    cl.forward(x[1])

    # 计算误差项
    sensitivityArray = np.ones(
        cl.stateList[-1].shape,
        dtype=np.float64
    )

    # 计算梯度
    cl.backward(sensitivityArray, ReluActivator())

    # 检查梯度
    epsilon = 10e-4
    for i in range(cl.W.shape[0]):
        for j in range(cl.W.shape[1]):
            cl.W[i, j] += epsilon
            cl.resetState()
            cl.forward(x[0])
            cl.forward(x[1])
            err1 = errorFunction(cl.stateList[-1])

            cl.W[i, j] -= 2 * epsilon
            cl.resetState()
            cl.forward(x[0])
            cl.forward(x[1])
            err2 = errorFunction(cl.stateList[-1])

            expectGrad = (err1 - err2) / (2 * epsilon)
            cl.W[i, j] += epsilon

            print(
                'weights(%d, %d): expected - actural %f - %f' % \
                    (
                        i, j,
                        expectGrad,
                        cl.gradient[i, j]
                    )
            )

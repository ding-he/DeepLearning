import numpy as np

class SigmoidActivator:
    '''
    sigmoid激活函数
    '''
    def forward(self, weightedInput):
        '''
        前向计算输出值
        '''
        return 1.0 / (1.0 + np.exp(-weightedInput))

    def backward(self, output):
        '''
        反向计算导数值
        '''
        return output * (1 - output)


class TanhActivator:
    '''
    tanh激活函数
    '''
    def forward(self, weightedInput):
        '''
        前向计算输出值
        '''
        return 2.0 / (1.0 + np.exp(-2 * weightedInput)) - 1.0

    def backward(self, output):
        '''
        反向计算导数值
        '''
        return 1 - output * output


class LstmLayer:
    '''
    长短时记忆网络层
    '''
    def __init__(self, inputWidth, stateWidth, learningRate):
        '''
        构造LSTM层
        inputWidth:     输入长度
        stateWidth:     记忆长度
        learningRate:   学习率
        '''
        self.inputWidth = inputWidth
        self.stateWidth = stateWidth
        self.learningRate = learningRate

        # 门的激活函数
        self.gateActivator = SigmoidActivator()
        # 输出的激活函数
        self.outputActivator = TanhActivator()

        # 当前时刻初始化为0
        self.times = 0

        # 每个时刻的状态向量c
        self.cList = self.initStateVector()
        # 每个时刻的输出向量h
        self.hList = self.initStateVector()
        # 每个时刻的遗忘门f
        self.fList = self.initStateVector()
        # 各个时刻的输入门i
        self.iList = self.initStateVector()
        # 各个时刻的输出门o
        self.oList = self.initStateVector()
        # 各个时刻的即时状态c~
        self.ctList = self.initStateVector()

        # 遗忘门的权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf = self.initWeightMat()
        # 输入门的权重矩阵Wih, Wix, 偏置项bi
        self.Wih, self.Wix, self.bi = self.initWeightMat()
        # 输出门的权重矩阵Woh, Wox, 偏置项bo
        self.Woh, self.Wox, self.bo = self.initWeightMat()
        # 单元状态的权重矩阵Wch, Wcx, 偏置项bc
        self.Wch, self.Wcx, self.bc = self.initWeightMat()

    def initStateVector(self):
        '''
        初始化保存状态的向量
        '''
        stateVectorList = []
        stateVectorList.append(np.zeros((self.stateWidth, 1)))
        return stateVectorList

    def initWeightMat(self):
        '''
        初始化权重矩阵
        '''
        Wh = np.random.uniform(
            -1e-4, 1e-4,
            (self.stateWidth, self.stateWidth)
        )
        Wx = np.random.uniform(
            -1e-4, 1e-4,
            (self.stateWidth, self.stateWidth)
        )
        b = np.zeros((self.stateWidth, 1))

        return Wh, Wx, b

    def forward(self, x):
        '''
        前向计算输出值
        '''
        self.times += 1

        # 遗忘门
        fg = self.calculateGate(
            x,
            self.Wfx, self.Wfh,
            self.bf,
            self.gateActivator
        )
        self.fList.append(fg)

        # 输入门
        ig = self.calculateGate(
            x,
            self.Wix, self.Wih,
            self.bi,
            self.gateActivator
        )
        self.iList.append(ig)

        # 输出门
        og = self.calculateGate(
            x,
            self.Wox, self.Woh,
            self.bo,
            self.gateActivator
        )
        self.oList.append(og)

        # 即时状态
        ct = self.calculateGate(
            x,
            self.Wcx, self.Wch,
            self.bc,
            self.outputActivator
        )
        self.ctList.append(ct)

        # 单元状态
        c = fg * self.cList[self.times - 1] + ig * ct
        self.cList.append(c)
        
        # 输出
        h = og * self.outputActivator.forward(c)
        self.hList.append(h)

    def calculateGate(self, x, Wx, Wh, b, activator):
        '''
        计算门输出
        '''
        # 上次的LSTM输出
        h = self.hList[self.times - 1]

        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)

        return gate

    def backward(self, x, deltaH, activator):
        '''
        反向计算误差项和梯度
        '''
        self.calculateDelta(deltaH, activator)
        self.calculateGradient(x)

    def calculateDelta(self, deltaH, activator):
        '''
        计算误差项
        '''
        # 初始化各个时刻的误差项
        self.deltaHList = self.initDelta()
        self.deltaOList = self.initDelta()
        self.deltaIList = self.initDelta()
        self.deltaFList = self.initDelta()
        self.deltaCtList = self.initDelta()

        # 保存从上一层传递下来的当前时刻的误差项
        self.deltaHList[-1] = deltaH

        # 迭代计算每个时刻的误差项
        for i in range(self.times, 0, -1):
            self.calculateDeltaSingle(i)

    def initDelta(self):
        '''
        初始化误差项
        '''
        deltaList = []
        for i in range(self.times + 1):
            deltaList.append(np.zeros(self.stateWidth, 1))
        
        return deltaList

    def calculateDeltaSingle(self, time):
        '''
        根据当前时刻的deltaH
        计算当前时刻的deltaF, deltaI
        deltaO, deltaCt, 以及上一时刻的deltaH
        '''
        # 获取当前时刻的前向计算的值
        ig = self.iList[time]
        og = self.oList[time]
        fg = self.fList[time]
        ct = self.ctList[time]
        c = self.cList[time]
        cPrev = self.cList[time - 1]
        tanhC = self.outputActivator.forward(c)
        deltaK = self.deltaHList[time]

        # 计算delta
        deltaO = (deltaK * tanhC * self.gateActivator.backward(og))
        deltaF = (deltaK * og * (1 - tanhC * tanhC) * cPrev * 
            self.gateActivator.backward(fg))
        deltaI = (deltaK * og * (1 - tanhC * tanhC) * ct * 
            self.gateActivator.backward(ig))
        deltaCt = (deltaK * og * (1 - tanhC * tanhC) * ig * 
            self.outputActivator.backward(ct))
        deltaHPrev = np.transpose(
            np.dot(np.transpose(deltaO), self.Woh) +
            np.dot(np.transpose(deltaI), self.Wih) +
            np.dot(np.transpose(deltaF), self.Wfh) +
            np.dot(np.transpose(deltaCt), self.Wch)
        )

        # 保存全部的delta
        self.deltaHList[time - 1] = deltaHPrev
        self.deltaFList[time] = deltaF
        self.deltaIList[time] = deltaI
        self.deltaOList[time] = deltaO
        self.deltaCtList[time] = deltaCt

    def calculateGradient(self, x):
        '''
        计算梯度
        '''
        # 初始化权重矩阵和偏置项
        self.WfhGrad, self.WfxGrad, self.bfGrad = self.initWeightGradMat()
        self.WihGrad, self.WixGrad, self.bfGrad = self.initWeightGradMat()
        self.WohGrad, self.WoxGrad, self.bfGrad = self.initWeightGradMat()
        self.WchGrad, self.WcxGrad, self.bfGrad = self.initWeightGradMat()

        # 计算上一次输出h的权重梯度
        for i in range(self.times, 0, -1):
            # 计算各个时刻的梯度
            (WfhGrad, bfGrad,
            WihGrad, biGrad,
            WohGrad, boGrad,
            WchGrad, bcGrad) = self.calculateGradient_t(i)

            # 实际梯度是各个时刻梯度之和
            self.WfhGrad += WfhGrad
            self.bfGrad += bfGrad
            self.WihGrad += WihGrad
            self.biGrad += biGrad
            self.WohGrad += WohGrad
            self.boGrad += boGrad
            self.WchGrad += WchGrad
            self.bcGrad += bcGrad

            print('------%d------' % i)
            print(WfhGrad)
            print(self.WfhGrad)

        # 计算本次输入x的权重梯度
        xt = np.transpose(x)
        self.WfxGrad = np.dot(self.deltaFList[-1], xt)
        self.WixGrad = np.dot(self.deltaIList[-1], xt)
        self.WoxGrad = np.dot(self.deltaOList[-1], xt)
        self.WcxGrad = np.dot(self.deltaCList[-1], xt)

    def initWeightGradMat(self):
        '''
        初始化权重矩阵
        '''
        WhGrad = np.zeros((self.stateWidth, self.stateWidth))
        WxGrad = np.zeros((self.stateWidth, self.inputWidth))
        bGrad = np.zeros((self.stateWidth, 1))

        return WhGrad, WxGrad, bGrad

    def calculateGradient_t(self, t):
        '''
        计算每个时刻t的权重梯度
        '''
        hPrev = np.transpose(self.hList[t - 1])
        WfhGrad = np.dot(self.deltaFList[t], hPrev)
        bfGrad = self.deltaFList[t]

        WihGrad = np.dot(self.deltaIList[t], hPrev)
        biGrad = self.deltaIList[t]

        WohGrad = np.dot(self.deltaOList[t], hPrev)
        boGrad = self.deltaOList[t]

        WchGrad = np.dot(self.deltaCtList[t], hPrev)
        bcGrad = self.deltaCtList[t]

        return WfhGrad, bfGrad, WihGrad, biGrad, \
            WohGrad, boGrad, WchGrad, bcGrad

    def update(self):
        '''
        梯度下降算法更新梯度
        '''
        self.Wfh -= self.learningRate * self.WfhGrad
        self.Wfx -= self.learningRate * self.WfxGrad
        self.bf -= self.learningRate * self.bfGrad

        self.Wih -= self.learningRate * self.WihGrad
        self.Wix -= self.learningRate * self.WixGrad
        self.bi -= self.learningRate * self.biGrad

        self.Woh -= self.learningRate * self.WohGrad
        self.Wox -= self.learningRate * self.WoxGrad
        self.bo -= self.learningRate * self.boGrad

        self.Wch -= self.learningRate * self.WchGrad
        self.Wcx -= self.learningRate * self.WcxGrad
        self.bc -= self.learningRate * self.bcGrad

    def resetState(self):
        '''
        重置状态
        '''
        self.times = 0

        # 初始化向量
        self.cList = self.initStateVector()
        self.hList = self.initStateVector()
        self.fList = self.initStateVector()
        self.iList = self.initStateVector()
        self.oList = self.initStateVector()
        self.ctList = self.initStateVector()


def getDataSet():
    '''
    获取数据集
    '''
    x = [np.array([[1], [2], [3]]),
        np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])

    return x, d


def gradientCheck():
    '''
    梯度检查
    '''
    # 误差函数, 取所有节点输出项之和
    errorFunction = lambda o: o.sum()

    lstm = LstmLayer(3, 2, 1e-3)

    # 计算forward输出值
    x, d = getDataSet()
    lstm.forward(x[0])
    lstm.forward(x[1])

    # 计算误差项
    sensitivityArray = np.ones(
        lstm.stateList[-1].shape,
        dtype=np.float64
    )

    # 计算梯度
    lstm.backward(x[1], sensitivityArray, ReluActivator())

    # 检查梯度
    epsilon = 10e-4
    for i in range(lstm.Wfh.shape[0]):
        for j in range(lstm.Wfh.shape[1]):
            lstm.Wfh[i, j] += epsilon
            lstm.resetState()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1 = errorFunction(lstm.stateList[-1])

            lstm.Wfh[i, j] -= 2 * epsilon
            lstm.resetState()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2 = errorFunction(lstm.stateList[-1])

            expectGrad = (err1 - err2) / (2 * epsilon)
            lstm.Wfh[i, j] += epsilon

            print(
                'weights(%d, %d): expected - actural %f - %f' % \
                    (
                        i, j,
                        expectGrad,
                        lstm.WfhGrad[i, j]
                    )
            )

    return lstm


if __name__ == '__main__':
    gradientCheck()

import numpy as np

class TreeNode:
    '''
    树节点
    '''
    def __init__(self, data, children = [], childrenData = []):
        '''
        构造树节点
        data:           数据
        children:       子节点
        childrenData:   子节点数据
        '''
        self.parent = None
        self.children = children
        self.childrenData = childrenData
        self.data = data

        # 对子节点添加父节点
        for child in children:
            child.parent = self


class RecursiveLayer:
    '''
    递归神经网络
    '''
    def __init__(self, nodeWidth, childCount, activator, learningRate):
        '''
        构造递归神经网络
        nodeWidth:          节点数据宽度
        childCount:         子节点数
        activator:          激活函数
        learningRate:       学习率
        '''
        self.nodeWidth = nodeWidth
        self.childCount = childCount
        self.activator = activator
        self.learningRate = learningRate

        # 权重数组
        self.W = np.random.uniform(
            -1e-4, 1e-4,
            (nodeWidth, nodeWidth * childCount)
        )

        # 偏置项
        self.b = np.zeros((nodeWidth, 1))

        # 递归神经网络生成树的根节点
        self.root = None
    
    def forward(self, *children):
        '''
        前向计算输出值
        '''
        childrenData = self.concatenate(children)
        parentData = self.activator.forward(
            np.dot(self.W, childrenData) + self.b
        )
        self.root = TreeNode(parentData, children, childrenData)

    def concatenate(self, treeNodes):
        '''
        将各个树节点中的数据拼接成一个长向量
        '''
        concat = np.zeros((0, 1))
        for node in treeNodes:
            concat = np.concatenate((concat, node.data))
        
        return concat

    def backward(self, parentDelta):
        '''
        BPTS反向传播计算误差值
        '''
        self.calculateDelta(parentDelta, self.root)
        self.WGrad, self.bGrad = self.calculateGradient(self.root)

    def calculateDelta(self, parentDelta, parent):
        '''
        计算每个节点的delta
        '''
        parent.delta = parentDelta

        # 如果有子节点
        if parent.children:
            childrenData = np.dot(
                np.transpose(self.W), parentDelta
            ) * (self.activator.backward(parent.childrenData))

            # slices = [(子节点编号, 子节点delta起始位置, 子节点delta结束位置)]
            slices = [
                (i, i * self.nodeWidth,
                (i + 1) * self.nodeWidth)
                for i in range(self.childCount)
            ]

            # 针对每个子节点, 递归调用calculateDelta函数
            for s in slices:
                self.calculateDelta(
                    childrenData[s[1]:s[2]], 
                    parent.children[s[0]]
                )

    def calculateGradient(self, parent):
        '''
        计算每个节点权重的梯度
        求和得到最终的梯度
        '''
        WGrad = np.zeros((
            self.nodeWidth,
            self.nodeWidth * self.childCount
        ))
        bGrad = np.zeros((self.nodeWidth, 1))

        if not parent.children:
            return WGrad, bGrad
        
        parent.WGrad = np.dot(
            parent.delta,
            np.transpose(parent.childrenData)
        )
        parent.bGrad = parent.delta

        WGrad += parent.WGrad
        bGrad += parent.bGrad

        # 迭代计算梯度
        for child in parent.children:
            W, b = self.calculateGradient(child)
            WGrad += W
            bGrad += b
        
        return WGrad, bGrad

    def update(self):
        '''
        梯度下降算法更新权重
        '''
        self.W -= self.learningRate * self.WGrad
        self.b -= self.learningRate * self.bGrad


def gradientCheck():
    '''
    梯度检查
    '''
    # 误差函数, 取所有节点输出项之和
    errorFunction = lambda o: o.sum()

    rnn = RecursiveLayer(2, 2, IndentityActivator(), 1e-3)

    # 计算forward输出值
    x, d = getDataSet()
    rnn.forward(x[0], x[1])
    rnn.forward(rnn.root, x[2])

    # 计算误差项
    sensitivityArray = np.ones(
        (rnn.nodeWidth, 1),
        dtype=np.float64
    )

    # 计算梯度
    rnn.backward(sensitivityArray)

    # 检查梯度
    epsilon = 10e-4
    for i in range(rnn.Wfh.shape[0]):
        for j in range(rnn.Wfh.shape[1]):
            rnn.Wfh[i, j] += epsilon
            rnn.resetState()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err1 = errorFunction(rnn.root.data)

            rnn.Wfh[i, j] -= 2 * epsilon
            rnn.resetState()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err2 = errorFunction(rnn.root.data)

            expectGrad = (err1 - err2) / (2 * epsilon)
            rnn.Wfh[i, j] += epsilon

            print(
                'weights(%d, %d): expected - actural %f - %f' % \
                    (
                        i, j,
                        expectGrad,
                        rnn.WGrad[i, j]
                    )
            )

    return rnn
        
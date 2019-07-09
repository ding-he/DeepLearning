import numpy as np

class ConvLayer:
    '''
    卷积层
    '''

    def __init__(
        self,
        inputWidth,
        inputHeight,
        channelNumber,
        filterWidth,
        filterHeight,
        filterNumber,
        zeroPadding,
        stride,
        activator,
        learningRate):

        '''
        构造卷积层
        inputWidth:         输入层的宽度
        inputHeight:        输入层的高度
        channelNumber:      输入层的维度
        filterWidth:        卷积模板的宽度
        filterHeight:       卷积模板的高度
        filterNumber:       卷积模板的数量
        zeroPadding:        补0的圈数
        stride:             卷积步长
        activator:          激活函数
        learningRate:       学习率
        '''
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.channelNumber = channelNumber
        self.filterWidth = filterWidth
        self.filterHeight = filterHeight
        self.filterNumber = filterNumber
        self.zeroPadding = zeroPadding
        self.stride = stride
        self.activator = activator
        self.learningRate = learningRate

        # 计算输出层的大小
        self.outputWidth = ConvLayer.calculateOutputSize(
            self.inputWidth,
            self.filterWidth,
            self.zeroPadding,
            self.stride
        )
        self.outputHeight = ConvLayer.calculateOutputSize(
            self.inputHeight,
            self.filterHeight,
            self.zeroPadding,
            self.stride
        )

        # 输出层矩阵
        self.outputArray = np.zeros((
            self.filterNumber,
            self.outputHeight,
            self.outputWidth
        ))

        # 本层的卷积模板
        self.filters = []
        for i in range(filterNumber):
            self.filters.append(
                Filter(filterWidth, filterHeight, channelNumber)
            )
    
    @staticmethod
    def calculateOutputSize(inputSize, filterSize, zeroPadding, stride):
        '''
        静态方法
        计算输出层的尺度
        '''
        return int((inputSize - filterSize + 2 * zeroPadding) / stride + 1)

    def forward(self, inputArray):
        '''
        前向计算
        计算输出值
        '''
        self.inputArray = inputArray

        # 在输入值上进行补0
        self.paddingInputArray = padding(inputArray, self.zeroPadding)

        # 计算每一个卷积模板对应的输出
        for i in range(self.filterNumber):
            filter = self.filters[i]
            conv(
                self.paddingInputArray,
                filter.getWeights,
                self.outputArray[i],
                self.stride,
                filter.getBias
            )

        # 对每一个输出元素操作
        elementWise(self.outputArray, self.activator.forward)

    def bpSensitivityMap(self, sensitivityArray, activator):
        '''
        计算传递到上一层的误差项
        sensitivityArray:       本层的误差项
        activator:              上一层的激活函数
        '''
        # 处理卷积步长, 对原始误差项进行扩展
        expandedArray = self.expandSensitivityMap(sensitivityArray)

        # full卷积, 对误差项进行zero padding
        expandedWidth = expandedArray.shape[2]
        zeroPadding = (self.inputWidth + self.filterWidth
            - 1 - expandedWidth) / 2
        paddedArray = padding(expandedArray, zeroPadding)

        # 初始化delta, 用于保存传递到上一层的误差项
        self.deltaArray = self.createDeltaArray()

        # 对于具有多个卷积模板的卷积层来说
        # 最终传递到上一层的误差项相当于所有
        # 卷积模板产生的误差项之和
        for i in range(self.filterNumber):
            filter = self.filters[i]

            # 将filter权重旋转180度
            flippedWeights = np.array(map(
                lambda x: np.rot90(x, 2),
                filter.getWeights()
            ))

            # 计算与一个filter对应的deltaArray
            deltaArray = self.createDeltaArray()
            for j in range(deltaArray.shape[0]):
                conv(
                    paddedArray[i],
                    flippedWeights[j],
                    deltaArray[d],
                    1, 0
                )
            
            self.deltaArray += deltaArray
        
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivativeArray = np.array(self.inputArray)
        elementWise(derivativeArray, activator.backward)

        self.deltaArray *= derivativeArray

    def expandSensitivityMap(self, sensitivityArray):
        '''
        将步长为S误差项还原成步长为1的误差项
        '''
        depth = sensitivityArray.shape[0]

        # 计算扩展后误差项的大小
        # 计算步长为1的误差项的大小
        expandedWidth = (self.inputWidth - self.filterWidth
            + 2 * self.zeroPadding + 1)
        expandedHeight = (self.inputHeight - self.filterHeight
            + 2 * self.zeroPadding + 1)

        # 构建新的误差项
        expandedArray = np.zeros((
            depth,
            expandedWidth,
            expandedHeight
        ))

        # 从原始的误差项拷贝数据
        for i in range(self.outputHeight):
            for j in range(self.outputWidth):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expandedArray[:, i_pos, j_pos] = \
                    sensitivityArray[:, i, j]
        
        return expandedArray

    def createDeltaArray(self):
        '''
        创建保存传递到上一层的误差项
        '''
        return np.zeros((
            self.channelNumber,
            self.inputHeight,
            self.inputWidth
        ))

    def bpGradient(self, sensitivityArray):
        '''
        计算梯度
        '''
        # 处理卷积步长, 对误差项进行扩展补0
        expandedArray = self.expandSensitivityMap(sensitivityArray)

        for i in range(self.filterNumber):
            # 计算每个权重的梯度
            filter = self.filters[i]
            for j in range(filter.weights.shape[0]):
                conv(
                    self.paddingInputArray[j],
                    expandedArray[i],
                    filter.weightsGrad[j],
                    1, 0
                )
            
            # 计算偏置项的梯度
            filter.biasGrad = expandedArray[i].sum()

    def update(self):
        '''
        根据梯度下降算法, 更新权重
        '''
        for filter in self.filters:
            filter.update(self.learningRate)


class Filter:
    '''
    卷积模板类
    保存卷积层参数和梯度
    使用梯度下降算法更新参数
    '''

    def __init__(self, width, height, depth):
        '''
        构造卷积模板
        width:      卷积模板的宽度
        height:     卷积模板的高度
        depth:      卷积模板的维度
        '''
        # 卷积模板权重参数
        self.weights = np.random.uniform(
            -1e-4, 1e-4,
            (depth, height, width)
        )

        # 偏置量
        self.bias = 0

        # 权重梯度
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = 0

    def __repr__(self):
        return 'filter weights: \n%s\nbias:\n%s' % \
            (repr(self.weights), repr(self.bias))

    def getWeights(self):
        '''
        返回权重矩阵
        '''
        return self.weights

    def getBias(self):
        '''
        返回偏置量
        '''
        return self.bias

    def update(self, learningRate):
        '''
        更新权重
        '''
        self.weights -= learningRate * self.weightsGrad
        self.bias -= learningRate * self.biasGrad


class ReluActivator:
    '''
    Relu激活函数
    '''
    def forward(self, weightedInput):
        '''
        前向计算
        计算输出值
        '''
        return max(0, weightedInput)

    def backward(self, output):
        '''
        反向计算
        计算导数值
        '''
        return 1 if output > 0 else 0


def elementWise(array, op):
    '''
    将元素进行迭代输出
    '''
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(inputArray, kernelArray, outputArray, stride, bias):
    '''
    计算卷积
    实现二维和三维的卷积运算
    '''
    channelNumber = inputArray.ndim
    outputWidth = outputArray.shape[1]
    outputHeight = outputArray.shape[0]
    kernelWidth = kernelArray.shape[-1]
    kernelHeight = kernelArray.shape[-2]

    # 计算卷积
    for i in range(outputHeight):
        for j in range(outputWidth):
            outputArray[i][j] = (
                getPatch(
                    inputArray,
                    i, j,
                    kernelWidth,
                    kernelHeight,
                    stride
                ) * kernelArray
            ).sum() + bias


def padding(inputArray, zeroPadding):
    '''
    为数组补0
    '''
    if zeroPadding == 0:
        return inputArray
    else:
        if inputArray.ndim == 3:
            inputWidth = inputArray.shape[2]
            inputHeight = inputArray.shape[1]
            inputDepth = inputArray.shape[0]

            # 生成全0数组
            paddingArray = np.zeros(
                inputDepth,
                inputHeight + 2 * zeroPadding,
                inputWidth + 2 * zeroPadding
            )

            # 复制源数组
            paddingArray[
                :,
                zp : zp + inputHeight,
                zp : zp + inputWidth
            ] = inputArray

            return paddingArray

        elif inputArray.ndim == 2:
            inputWidth = inputArray.shape[1]
            inputHeight = inputArray.shape[0]

            # 生成全0数组
            paddingArray = np.zeros(
                inputHeight + 2 * zeroPadding,
                inputWidth + 2 * zeroPadding
            )

            # 复制源数组
            paddingArray[
                zp : zp + inputHeight,
                zp : zp + inputWidth
            ] = inputArray

            return paddingArray


class MaxPoolingLayer:
    '''
    max pooling降采样层
    '''
    def __init__(
        self,
        inputWidth,
        inputHeight,
        channelNumber,
        filterWidth,
        filterHeight,
        stride):
        '''
        构造max pooling层
        inputWidth:         输入层的宽度
        inputHeight:        输入层的高度
        channelNumber:      输入层的维度
        filterWidth:        卷积模板的宽度
        filterHeight:       卷积模板的高度
        stride:             卷积步长
        '''
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.channelNumber = channelNumber
        self.filterWidth = filterWidth
        self.filterHeight = filterHeight
        self.stride = stride

        # 计算输出层的大小
        self.outputWidth = (inputWidth - filterWidth) / stride + 1
        self.outputHeight = (inputHeight - filterHeight) / stride + 1

        # 初始化输出矩阵
        self.outputArray = np.zeros(
            self.channelNumber,
            self.outputHeight,
            self.outputWidth
        )

    def forward(self, inputArray):
        '''
        前向计算输出值
        '''
        for k in range(self.channelNumber):
            for i in range(self.outputHeight):
                for j in range(self.outputWidth):
                    self.outputArray[k, i, j] = (
                        getPatch(
                            inputArray[k],
                            i, j,
                            self.filterWidth,
                            self.filterHeight,
                            self.stride
                        ).max()
                    )

    def backward(self, inputArray, sensitivityArray):
        '''
        反向计算梯度值
        '''
        self.deltaArray = np.zeros(inputArray.shape)

        for k in range(self.channelNumber):
            for i in range(self.outputHeight):
                for j in range(self.outputWidth):
                    patchArray = getPatch(
                        inputArray[d],
                        i, j,
                        self.filterWidth,
                        self.filterHeight,
                        self.stride
                    )

                    k, l = getMaxIndex(patchArray)

                    self.deltaArray[
                        d,
                        i * self.stride + k,
                        j * self.stride + l
                    ] = sensitivityArray[k, i, j]

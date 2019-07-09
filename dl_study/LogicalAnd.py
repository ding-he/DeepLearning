class Sensor:
    def __init__(self, inputNums, activeFunction):
        # 初始化激活函数
        self.activeFunction = activeFunction

        # 初始化参数个数
        self.inputNums = inputNums

        # 初始化输入权重为0
        self.weights = [0.0 for x in range(inputNums)]

        # 初始化偏量为0
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbais\t:%s' % (self.weights, self.bias)

    # 输出预测值
    def predict(self, inputVector):
        output = 0.0
        for i in range(self.inputNums):
            output += inputVector[i] * self.weights[i]
        return self.activeFunction(output + self.bias)

    def train(self, inputVectors, labels, iterNums, rate):
        for i in range(iterNums):
            self._one_train(inputVectors, labels, rate)

    def _one_train(self, inputVectors, labels, rate):
        # 把每个数据都进行一次预测
        # 再把结果与实际值的偏差进行学习
        samples = zip(inputVectors, labels)

        for (inputVector, label) in samples:
            # 预测
            output = self.predict(inputVector)

            # 计算与实际值的差值
            delta = label - output

            # 更新权重与偏移值
            for i in range(self.inputNums):
                self.weights[i] += rate*delta*inputVector[i]
            self.bias += rate*delta

# 定义激活函数
def step(x):
    return 1 if x > 0 else 0

# 训练数据集
def trainSensorDataSets():
    vectors = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]

    return vectors, labels

# 训练与感知器
def trainAndSensor():
    andSensor = Sensor(2, step)
    
    # 获取训练数据集
    vectors, labels = trainSensorDataSets()

    # 训练感知器
    # 迭代10次, 学习率为0.1
    andSensor.train(vectors, labels, 10, 0.1)

    return andSensor

if __name__ == '__main__':
    # 训练感知器
    andSensor = trainAndSensor()

    print(andSensor)

    # 测试数据
    print('0 && 0 => %d' % andSensor.predict([0, 0]))
    print('0 && 1 => %d' % andSensor.predict([0, 1]))
    print('1 && 0 => %d' % andSensor.predict([1, 0]))
    print('1 && 1 => %d' % andSensor.predict([1, 1]))


    
from LogicalAnd import Sensor

# 线性单元 梯度下降
class LinearUint(Sensor):
    def __init__(self, inputNums, activeFunction):
        super().__init__(inputNums, activeFunction)

# 线性单元模型
def linearModule(x):
    return x

# 训练数据集
def trainSensorDataSets():
    vectors = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]

    return vectors, labels

# 训练线性单元
def trainLinearUint():
    uint = LinearUint(1, linearModule)

    vectors, labels = trainSensorDataSets()

    # 训练线性单元
    # 采用梯度下降算法寻找最优值
    uint.train(vectors, labels, 10, 0.01)

    return uint

if __name__ == '__main__':

    # 训练线性单元
    uint = trainLinearUint()

    print(uint)

    # 测试数据集
    print('Work 3.4 years, monthly salary = %.2f' % uint.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % uint.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % uint.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % uint.predict([6.3]))


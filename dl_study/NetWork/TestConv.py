import numpy as np
import Conv


def init_test():
    '''
    初始化测试
    '''
    # 输入层
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]]
    )

    # 输出层
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]]
    )

    # 构建卷积层
    cl = Conv.ConvLayer(
        5, 5, 3, 3, 3, 2, 1, 2,
        Conv.ReluActivator(),
        0.001
    )

    # 初始化卷积模板
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]],
        dtype=np.float64
    )
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]],
        dtype=np.float64
    )

    return a, b, cl


def gradientCheck():
    '''
    梯度检查
    '''
    # 设计一个误差函数, 取所有节点输出项之和
    errorFunction = lambda o: o.sum()

    # 计算前向输出值
    a, b, cl = init_test()
    cl.forward(a)

    # 求取误差项, 为一个全1数组
    sensitivityArray = np.ones(cl.outputArray.shape, dtype=np.float64)

    # 反向计算梯度
    cl.backward(a, sensitivityArray, Conv.ReluActivator())

    # 检查梯度
    epsilon = 10e-4

    for k in range(cl.filters[0].weightsGrad.shape[0]):
        for i in range(cl.filters[0].weightsGrad.shape[1]):
            for j in range(cl.filters[0].weightsGrad.shape[2]):
                cl.filters[0].weights[k, i, j] += epsilon
                cl.forward(a)
                err1 = errorFunction(cl.outputArray)

                cl.filters[0].weights[k, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = errorFunction(cl.outputArray)

                expectGrad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[k, i, j] += epsilon

                print(
                    'weights(%d, %d, %d): expected - actural %f - %f') % \
                    (
                        k, i, j,
                        expectGrad,
                        cl.filters[0].weightsGrad[k, i, j]
                    )
                )

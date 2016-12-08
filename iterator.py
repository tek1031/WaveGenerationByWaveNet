import chainer
import random
import numpy as np

class Iterator():
    def __init__(self, batchSize, data, answer, shuffle = True, useGPU = True):
        self.shuffle = shuffle
        self.batchSize = batchSize
        self.newEpoch = True
        self.data = data
        self.answer = answer
        self.pos = 0
        self.useGPU = useGPU

        if useGPU:
            self.data = chainer.cuda.to_gpu(data)
            self.answer = chainer.cuda.to_gpu(answer)


    def Next(self):
        xp = chainer.cuda.cupy if self.useGPU else np
        if self.newEpoch:
            self.newEpoch = False
            self.pos = 0
            self.order = np.arange(len(self.data))
            if self.shuffle:
                random.shuffle(self.order)

        N = len(self.data)
        n = min(self.batchSize, (N - self.pos))
        shape = self.data[0].shape
        shape = (n,) + shape
        x = xp.zeros(shape, dtype='f')
        shape = self.answer[0].shape
        shape = (n,) + shape
        t = xp.zeros(shape, dtype = 'i')
        for i in range(n):
            idx = self.order[self.pos + i]
            x[i] = self.data[idx]
            t[i] = self.answer[idx]
        self.pos += n
        if self.pos == len(self.data):
            self.newEpoch = True
        variable = chainer.Variable(x)
        return variable, t

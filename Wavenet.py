import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Link, Chain, ChainList
import chainer.cuda

class WaveBlock(Chain):
    def __init__(self, n_in_channel, n_skip_channel, n_dilate, useGPU):
        dilatedConvTan = L.DilatedConvolution2D(None, n_skip_channel, (1,2), dilate = n_dilate)
        dilatedConvSig = L.DilatedConvolution2D(None, n_skip_channel, (1,2), dilate = n_dilate)
        conv = L.Convolution2D(n_skip_channel, n_in_channel, ksize = 1)
        super(WaveBlock, self).__init__(
            dilatedConvTan = dilatedConvTan,
            dilatedConvSig = dilatedConvSig,
            conv = conv
        )
        self.useGPU = useGPU
    
    def __call__(self, x):
        xTan = self.dilatedConvTan(x)
        xTan = F.tan(xTan)
        xSig = self.dilatedConvSig(x)
        xSig = F.sigmoid(xSig)

        h = xTan*xSig
        skip = self.conv(h)

        s = x.shape
        xp = chainer.cuda.cupy if self.useGPU else np
        zero = xp.zeros((s[0], s[1], s[2], s[3]-skip.shape[3]), dtype = 'f')
        skip = F.concat((skip, zero), axis = 3)

        next = skip + x
        return next, skip

class Wavenet(Chain):
    def __init__(self, resolution, n_stack, n_dilateStack, n_in_channel, n_skip_channel, useGPU):
        firstConv = L.Convolution2D(None, n_in_channel, ksize=1)
        wn = []
        for s in range(n_stack):
            for d in range(n_dilateStack):
                wn.append(WaveBlock(n_in_channel, n_skip_channel, 2**d, useGPU))
        
        lastConv0 = L.Convolution2D(n_skip_channel, n_skip_channel, ksize = 1)
        lastConv1 = L.Convolution2D(n_skip_channel, n_skip_channel, ksize = 1)
        linear = L.Linear(None, resolution)
        super(Wavenet, self).__init__(
            firstConv = firstConv,
            waveBlocks = ChainList(*wn),
            lastConv0 = lastConv0,
            lastConv1 = lastConv1,
            linear = linear
        )

    def __call__(self, x, t):
        output = self.forward(x)
        loss = F.softmax_cross_entropy(output, t)
        print loss.data
        return loss
    
    def forward(self, x):
        x = self.firstConv(x)
        output = 0
        for i in range(len(self.waveBlocks)):
            x, skip = self.waveBlocks[i](x)
            output += skip

        output = F.relu(output)
        output = self.lastConv0(output)
        output = F.relu(output)
        output = self.lastConv1(output)
        output = self.linear(output)
        return output



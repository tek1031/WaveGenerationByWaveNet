import Wavenet
import numpy as np
import chainer
import math
from chainer import training
from chainer.training import extensions
import numpy as np
import chainer.cuda
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cupy
import wave
import sys
import math
import array
import iterator

MU = 256

def sgn(i):
    if i == 0:
        return 0
    return 1 if i>0 else -1

def normalize(x):
    min = np.min(x)
    x -= min
    max = np.max(x)
    x /= max
    x*=2.0
    x-=1.0
    return x, max, min

def deNormalize(x, max, min):
    x+=1.0
    x/=2.0
    x*= max
    x+=min
    return x

def quantizate(x, r):
    x = int(x*(r-1)/2)
    x /= ((r-1)/2.0)
    if x>=1.0 or x<=-1.0:
        print "???", x
    return x

def mu_law(v, mu):
    if math.fabs(v) >1.0:
        print "????", math.fabs(v)
    m = math.log(1 + mu * math.fabs(v)) / math.log(1. + mu)    
    return sgn(v)*m

def mu_law_decode(v, mu):
    m = (1. / mu) * ((1. + mu)**math.fabs(v) - 1.)
    return sgn(v)*m

def softMax(v):
    s = xp.sum(xp.exp(v))
    return xp.exp(v) / s

def saveSound(v, samplingRate, fileName):
    w = wave.Wave_write(fileName)
    w.setparams((
        1,                        # channel
        2,                        # byte width
        samplingRate,                    # sampling rate
        len(v),                 # number of frames
        "NONE", "not compressed"  # no compression
    ))
    w.writeframes(array.array('h', v).tostring())
    w.close()
    print "save :", fileName

#file_name = "880Hz.WAV"
#file_name = "0321.wav"
file_name = "Data/test2.wav"
wave_file = wave.open(file_name,"r") #open

music = wave_file.readframes(wave_file.getnframes()) 
music = np.frombuffer(music, dtype= "int16")
music = np.asarray(music, dtype = 'f')
rawMusic = np.copy(music)

TEST = 10
FLAME_RATE = 44100*2
music = music[::TEST]
music = music[len(music)*1/2:len(music)*2/2]

print "len Wav:", len(music)
saveSound(music, FLAME_RATE/TEST, "raw.wav")
music, max, min = normalize(music)
for i in range(len(music)):
    music[i] = mu_law(music[i], MU)

N_OUTPUT = 100
R = 64
L = 2**10
useGPU = True
model = Wavenet.Wavenet(N_OUTPUT,R,1,10,30,30,useGPU)
if useGPU:
    model.to_gpu()
xp = np

hoge = []
data = []
answers = []

pos = 0
log = 0

quantizedMusic = np.zeros((R, 1, len(music)), dtype = 'f')
for i in range(len(music)):
    v = (music[i] + 1.0) / 2.0
    idx = int(v * (R-1))
    quantizedMusic[idx][0][i] = 1.0

while pos + L + N_OUTPUT < len(music):
    currentLoad =  int((pos + L + N_OUTPUT) / float(len(music)) * 100)
    if currentLoad -10 >= log:
        log = currentLoad
        print currentLoad, "%"
        break
    x = quantizedMusic[:, :, pos: pos+L]
    """
    for i in range(L):
        v = (music[pos+i] + 1.0) / 2.0
        idx = int(v * (R-1))
        x[idx][0][i] = 1.0
    """
    answer = xp.zeros((N_OUTPUT), dtype = 'i')
    for i in range(N_OUTPUT):
        v = (music[pos+L+i] + 1.0) / 2.0
        idx = int(v * (R-1))
        answer[i] = idx
        hoge.append(idx)
    data.append(x)
    answers.append(answer)
    pos += N_OUTPUT

print "NUM DATA:", len(data)
trainIter = iterator.Iterator(10, data, answers, True)
optimizer = chainer.optimizers.Adam(alpha = 0.001)
optimizer.setup(model)

for i in range(10):
    print "EPOCH :", i
    while True:
        model.cleargrads()
        x, t =  trainIter.Next()
        loss = model(x, t)
        loss.backward()
        optimizer.update()
        if trainIter.newEpoch:
            break

d = data[0]
d = d.reshape(((1,) + d.shape))
if useGPU:
    d = chainer.cuda.to_gpu(d)
res = []
for i in range(2000):
    print i
    
    y = model.forward(d)
    next = np.zeros((R, 1, N_OUTPUT), dtype = 'f')
    for i in range(N_OUTPUT):
        idx = int(xp.argmax(y[i].data[0]))
        res.append(idx)
        next[idx][0][i] = 1.0
    next = chainer.cuda.to_gpu(next)
    d[:, :, :, :-N_OUTPUT] = d[:, :, :, N_OUTPUT:]
    d[:, :, :, L-N_OUTPUT:] = next

for i in range(len(res)):
    res[i] = res[i] / float(R-1)
    res[i] *= 2
    res[i] -= 1.0
res = np.asarray(res, dtype = "f")

for i in range(len(res)):
    res[i] = mu_law_decode(res[i], MU)

for i in range(len(res)):
    res[i] = deNormalize(res[i], max, min)

print len(res)
print res
saveSound(res, FLAME_RATE/TEST, "test0.wav")

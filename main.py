import Wavenet
import numpy as np
import chainer
import math
from chainer import training
from chainer.training import extensions
import numpy as np
import chainer.cuda
import matplotlib.pyplot as plt
import cupy


import wave
import sys
import math
import array

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
    

file_name = "0321.wav"
wave_file = wave.open(file_name,"r") #Open


print wave_file.getnchannels()
print wave_file.getframerate()
print wave_file.getnframes() 
print wave_file.getparams()

x = wave_file.readframes(wave_file.getnframes()) 
x = np.frombuffer(x, dtype= "int16")
x = np.asarray(x, dtype = 'f')


plt.plot(x[1000:1100])
x, max, min = normalize(x)

for i in range(len(x)):
    x[i] = mu_law(x[i], 256)
    x[i] = quantizate(x[i], 256)

for i in range(len(x)):
    x[i] = mu_law_decode(x[i], 256)
x = deNormalize(x, max, min)
plt.plot(x[1000:1100])

w = wave.Wave_write("test.wav")
w.setparams((
    1,                        # channel
    2,                        # byte width
    48100,                    # sampling rate
    len(x),                 # number of frames
    "NONE", "not compressed"  # no compression
))
w.writeframes(array.array('h', x).tostring())
w.close()



"""
R = 32
L = 256
useGPU = True
model = Wavenet.Wavenet(R, 3,3,10,10,useGPU)
if useGPU:
    xp = chainer.cuda.cupy
    model.to_gpu()
else:
    xp = np
trainData = []

hoge = []
for i in range(1000):
    x = xp.zeros((R, 1, L), dtype = 'f')
    freq = 2*math.pi / 30
    rad = i*freq
    for j in range(L):
        v = (math.sin(rad) + 1.0) / 2.0
        idx = int(v * R)
        x[idx][0][j] = 1.0
        rad += freq
        hoge.append(v)
    v = (math.sin(rad) + 1.0) / 2.0
    idx = int(v * R)
    trainData.append((x, xp.asarray(idx, dtype = 'i')))

plt.plot(hoge[:100])
#plt.show()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
train_iter = chainer.iterators.SerialIterator(trainData, 128)
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (100, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.ProgressBar())
trainer.run()

"""
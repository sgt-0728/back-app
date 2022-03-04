#make NeuralNetwork
#load mnist
#forward processing
#find loss function
import numpy as np
import csv
import pandas as pd
from neural import *
import matplotlib.pyplot as plt

loss_list = []

lists = {}
lists['train_loss'] = []
lists['test_loss'] = []
lists['epoch'] = []


for k in range(1):
    NN = NeuralNetwork(1, 7, 1)
    data = func_sin()
    x, t = data.train()
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)

    w = NN.first_params()
    v = NN.first_v()
    H = NN.first_H()

    lr = 1
    u = 0.95
    e = 1e-6
    trials = 100000

    for i in range(trials):
        w1 = w + u * v
        loss, g1, w1 = NN.NN(x, t, w1)
        lr = NN.line_search(x, t, w1, H, g1)

        v = NN.update_v(u, v, lr, H, g1)
        w = NN.update_w(w, v)
        w2 = w

        loss, g2, w2 = NN.NN(x, t, w2)
        loss_list.append(loss)

        H = NN.update_H(H, w1, w2, g1, g2)

        g_norm = np.amax(np.abs(g2))
        if g_norm <=e:
            break

        if i % 1000 == 0:
            print(loss)

    lists['train_loss'].append(loss*1000)
    lists['epoch'].append(i)

    x_test, t_test = data.test()
    x_test = x_test.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    loss, grads, w = NN.NN(x_test, t_test, w)
    lists['test_loss'].append(loss*1000)
    #print(loss)

print(lists)

df = pd.DataFrame(loss_list)
#df = pd.DataFrame([lists['train_loss'],lists['test_loss'],lists['epoch']])

df.to_csv("out.csv")



#print(loss)
markers = {'loss'}
x = np.arange(len(loss_list))
plt.loglog(x, loss_list, label='loss')

plt.xlabel("trial_number")
plt.ylabel("loss")
plt.plot(0, np.max(loss_list))
plt.legend(loc='upper right')
plt.show()

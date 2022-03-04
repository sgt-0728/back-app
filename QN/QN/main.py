#make NeuralNetwork
#load mnist
#forward processing
#find loss function
import numpy as np
import pandas as pd
from neural import *
import matplotlib.pyplot as plt


loss_list = []

NN = NeuralNetwork(1, 7, 1)
data = func_sin()
x, t = data.data()
x = x.reshape(-1, 1)
t = t.reshape(-1, 1)
w = NN.first_params()
H = NN.first_H()

lr = 1
e = 1e-6

trials = 100000
loss, grads, w = NN.NN(x, t, w)

for i in range(trials):

    lr = NN.line_search(x, t, w, lr, H, grads)

    w1 = w
    g1 = grads

    v = NN.update_v(lr, H, grads)
    w = NN.update_w(w, v)


    loss, grads, w = NN.NN(x, t, w)
    loss_list.append(loss)

    w2 = w
    g2 = grads

    H = NN.update_H(H, w1, w2, g1, g2)


    g_norm = np.amax(np.abs(grads))
    #g_norm = np.linalg.norm(grads)
    if g_norm <=e:
        break


    if i % 1000 == 0:
        print(loss)

print(i)
df = pd.DataFrame(loss_list)
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

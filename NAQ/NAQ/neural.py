import sys,os
import numpy as np
from collections import OrderedDict

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #np.random.seed(100) #乱数を固定
        self.params = {}
        self.params['W1'] =np.ones([input_size, hidden_size])#np.random.uniform(-0.5,0.5,[input_size, hidden_size])
        self.params['b1'] =np.ones([1, hidden_size])#np.random.uniform(-0.5,0.5,[1, hidden_size])
        self.params['W2'] =np.ones([hidden_size, output_size])#np.random.uniform(-0.5,0.5,[hidden_size, output_size])
        self.params['b2'] =np.ones([1, output_size])#np.random.uniform(-0.5,0.5,[1, output_size])

        N = input_size * hidden_size + hidden_size * output_size + hidden_size + output_size

        self.H = np.eye(N)
        self.w = np.random.uniform(-0.5,0.5,[N, 1])#np.ones([N, 1])

        self.grads = {}

        self.v = {}
        self.v['W1'] =np.zeros([input_size, hidden_size])# np.random.uniform(-0.5,0.5,[input_size, hidden_size])
        self.v['b1'] =np.zeros([1,hidden_size])# np.random.uniform(-0.5,0.5,[hidden_size])
        self.v['W2'] =np.zeros([hidden_size, output_size])#np.random.uniform(-0.5,0.5,[hidden_size, output_size])
        self.v['b2'] =np.zeros([1,output_size]) #np.random.uniform(-0.5,0.5,[output_size])

    def first_params(self):
        return self.w

    def first_v(self):
        v = np.concatenate([self.v['W1'].reshape(-1, 1), self.v['b1'].reshape(-1, 1), self.v['W2'].reshape(-1, 1), self.v['b2'].reshape(-1, 1)])
        return v

    def first_H(self):
        return self.H

    def setting_layers(self, params):
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine1(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid'] = Sigmoid()
        self.layers['Affine2'] = Affine2(self.params['W2'], self.params['b2'])

        self.lastLayer = IdentityWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.lastLayer.forward(x, t)
        return y

    def backward(self, dout):
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

    def gradient(self):
        self.grads['W1'] = self.layers['Affine1'].dW
        self.grads['b1'] = self.layers['Affine1'].db
        self.grads['W2'] = self.layers['Affine2'].dW
        self.grads['b2'] = self.layers['Affine2'].db

        grads = np.concatenate([self.grads['W1'].reshape(-1, 1), self.grads['b1'].reshape(-1, 1), self.grads['W2'].reshape(-1, 1),  self.grads['b2'].reshape(-1, 1)])

        return grads

    def getting_loss(self, x, t, params):
        self.setting_layers(params)
        dout = self.predict(x)
        loss = self.loss(dout, t)
        return dout, loss

    def getting_grads(self, dout):
        self.backward(dout)
        grads = self.gradient()
        return grads

    def NN(self, x, t, w):
        params = self.split_params(w)
        dout, loss = self.getting_loss(x, t, params)
        grads = self.getting_grads(dout)
        w = self.conect_params()
        return loss, grads, w

    def conect_params(self):
        x = np.concatenate([self.params['W1'].reshape(-1, 1), self.params['b1'].reshape(-1, 1), self.params['W2'].reshape(-1, 1),  self.params['b2'].reshape(-1, 1)])
        return x

    def split_params(self, w):

        W1 = self.input_size * self.hidden_size
        W2 = self.hidden_size * self.output_size
        b1 = self.hidden_size
        b2 = self.output_size

        a = W1
        b = W1 + b1
        c = W1 + W2 + b1
        d = W1 + W2 + b1 + b2

        params = np.split(w, [a, b, c, d])

        #params['W1'].reshape([self.input_size, self.hidden_size])
        #params['b1'].reshape([1, self.hidden_size])
        #params['W2'].reshape([self.hidden_size, self.output_size])
        #params['b2'].reshape([1, self.output_size])

        i = 0
        for key in self.params.keys():
            self.params[key] = params[i].T
            i = i + 1
        self.params['W2'] = self.params['W2'].T
        return self.params


    def update_v(self, u, v, lr, H, g1):
        a = np.dot(H, g1)
        v = u * v - lr * a
        return v

    def update_w(self, w, v):
        w = w + v
        return w

    def line_search(self, x, t, w1, H, g1):
        lr = 1
        g = - np.dot(H, g1)
        for i in range(4):
            wl = w1 + lr * g
            E1, A , B = self.NN(x, t, w1)
            E2, C, D = self.NN(x, t, wl)

            L = E2
            R = E1 + 0.001 * lr * np.dot(g1.T, g)

            if L <= R:
                break
            lr *= 0.5

        return lr

    def update_H(self, H, w1, w2, g1, g2):
        g_norm = np.amax(np.abs(g1))
        s = w2 - w1
        y = g2 - g1

        p_q = np.dot(s.T, y)
        if g_norm > 1e-2:
            const = 2.0
        else:
            const = 100.0
        if p_q < 0:
            p_p = np.dot(s.T, s)
            zeta = const - (p_q / (p_p * g_norm))
        else:
            zeta = const
        y = y + zeta * g_norm * s

        a = np.dot(H, y)
        b = np.dot(s.T, y)
        c = np.dot(y.T, H)
        d = np.dot(s, s.T)

        H = H - (np.dot(a, s.T) + np.dot(s, a.T))/b + (1 + np.dot(c, y)/b) * d/b
        return H






#optimizer
    def SG(self, w, grads):
        lr = 0.01
        w -= lr * grads
        #for key in params.keys():
        #    self.params[key] -= lr * grads[key]
        return self.params

    def NAG(self, w, grads):
        lr = 0.01
        u = 0.95
        for key in self.grads.keys():
            self.v[key] = u * self.v[key] - lr * self.grads[key]
        v = np.concatenate([self.v['W1'].T, self.v['b1'].T, self.v['W2'],  self.v['b2']])

        w -= lr * grads
        w += u * v
        return w

class Affine1:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0,keepdims=True)
        return dx

class Affine2:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0,keepdims=True)
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = self.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

    def sigmoid(self, x):
        out = 1/(1+np.exp(-x))
        return out

class IdentityWithLoss:
    def __init__(self):
        self.loss = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        t = t.reshape(-1, 1)
        self.loss = 0.5*np.sum((t-x)**2)/t.shape[0]
        return self.loss

    def backward(self, dout):
        dx = (dout - self.t)/self.t.shape[0]
        return dx

class func_sin:
    def __init__(self):
        self.x = np.linspace(-4, 4, 400)
        self.x_test = np.random.uniform(-4, 4, 10000)

    def train(self):
        self.t = self.function(self.x)
        self.t = self.normalizeData(self.t).reshape(-1,1)
        self.x = self.normalizeData(self.x).reshape(-1,1)
        return self.x, self.t

    def test(self):
        self.t_test = self.function(self.x_test)
        self.t_test = self.normalizeData(self.t_test).reshape(-1,1)
        self.x_test = self.normalizeData(self.x_test).reshape(-1,1)
        return self.x_test, self.t_test

    def function(self, x):
        y = 1 + (x + 2*x*x)*np.sin(-x*x)
        return y

    def normalizeData(self,data):
        a = -1
        b = 1
        A = min(data)
        B = max(data)
        X = a + (data - A) * (b - a) / (B - A)
        return X

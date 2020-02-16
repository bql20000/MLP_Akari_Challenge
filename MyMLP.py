import numpy as np


class MyMLP(object):
    # Z[i] = dot(W[i], X[i])        (inactivated)   (save this for back-propagation calculation)
    # A[i]: sigmoid(Z[i])      (activated)
    # X = (n_feature + bias, n_samples)  --> (dim+1, N)
    # y = (10, n_samples) --> one-hot coding
    # N: number of samples
    # C: number of classes
    # W[i]: weight matrix in i_th layer

    def __init__(self, hidden_layer_size=(1, ), activation='sigmoid', learning_rate=0.1, A=None, Z=None,
                 W=None, X=None, Y=None, W_old=None, C=None, N=None, max_iter=100, momentum=0.9, loss_info=None):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.A = A
        self.Z = Z
        self.W = W
        self.C = C
        self.N = N
        self.X = X
        self.Y = Y
        self.W_old = W_old
        self.max_iter = max_iter
        self.momentum = momentum
        self.loss_info = loss_info

    def add_bias(self, data):
        return np.concatenate((np.ones((1, data.shape[1]))/self.C, data), axis=0)    # bias = first row

    def setup_parameters(self, images, labels):
        self.A = []
        self.Z = []
        self.W = []
        self.W_old = []
        self.loss_info = []
        #print(np.asarray(self.A).shape)
        self.N = images.shape[0]
        self.X = self.add_bias(images.T)           # transpose images and add bias (N x D) --> (N+1 x D)
        self.Y = np.zeros((self.C, self.N))
        for i in range(self.N):
            self.Y[labels[i]][i] = 1

        #hidden layers && weight matrices
        pre_layer_size = self.X.shape[0]
        for size in self.hidden_layer_size:
            self.A.append(np.zeros((size, self.N)))
            self.W.append(1 * (np.random.random((pre_layer_size, size)) - 0.5))
            pre_layer_size = size

        #ouput layers
        self.A.append(np.zeros((self.C, self.X.shape[1])))
        self.W.append(1 * (np.random.random((pre_layer_size, self.C)) - 0.5))

        #other parameters
        self.Z = self.A
        self.W_old = self.W

    def print(self):
        layers_size = self.hidden_layer_size.__add__((self.C,))
        print("MLP [layers= " + str(layers_size) + ", activation= " + self.activation + "]")

    def derivative_f(self, Z):
        if self.activation == 'sigmoid':
            return Z / ((1+Z)**2)
        if self.activation == 'relu':
            res = np.ones(Z.shape)
            res[Z <= 0] = 0
            return res
        return TypeError

    def softmax(self, Z):
        #print("dm: ",np.amin(Z), np.amax(Z))
        A = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return A / np.sum(A, axis=0)

    def loss(self):
        #print("-->", np.amin(self.A[-1]))
        return -np.sum(self.Y * np.log(self.A[-1])) / self.N   # becareful with overflow

    def activate_function(self, Z):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-Z))
        if self.activation == 'relu':
            return np.maximum(Z, 0)

    def forward_propagation(self):
        L = len(self.hidden_layer_size) + 1
        for i in range(L):
            if i == 0:
                self.Z[i] = np.dot(self.W[i].T, self.X)
            else:
                self.Z[i] = np.dot(self.W[i].T, self.A[i-1])

            if i == L-1:
                self.A[i] = self.softmax(self.Z[i])
            else:
                self.A[i] = self.activate_function(self.Z[i])

    def update_W(self, l, dWl):
        W_new = self.W[l] - self.learning_rate * dWl + self.momentum * (self.W[l] - self.W_old[l])
        self.W_old[l] = self.W[l]
        self.W[l] = W_new

    def back_propagation(self):
        L = len(self.hidden_layer_size)
        E = (self.A[-1] - self.Y) / self.N
        dWL = np.dot(self.A[L-1], E.T)
        self.update_W(L, dWL)
        for l in range(len(self.hidden_layer_size)-1, -1, -1):
            E = np.dot(self.W[l+1], E) * self.derivative_f(self.Z[l])
            dWl = np.zeros((1, 1))       #todo: ???
            if l == 0:
                dWl = np.dot(self.X, E.T)
            else:
                dWl = np.dot(self.A[l-1], E.T)
            self.update_W(l, dWl)
        return 0

    def fit(self, images, labels):
        self.setup_parameters(images, labels)
        self.loss_info = []
        for i in range(self.max_iter):
            #print(self.W[1].shape)
            self.forward_propagation()
            if i % (self.max_iter / 10) == 0:
                print("Iteration %d: loss = %.4f" % (i, self.loss()))
                self.loss_info.append(self.loss())
            self.back_propagation()

    def get_loss_info(self):
        return np.asarray(self.loss())

    def predict(self, test):
        Z = self.add_bias(test.T)
        L = len(self.hidden_layer_size) + 1
        for i in range(L):
            Z = np.dot(self.W[i].T, Z)
            if i == L - 1:
                Z = self.softmax(Z)
            else:
                Z = self.activate_function(Z)
        return np.argmax(Z, axis=0)

    #predict test and check with correct result. Return (the number of correct answer / total) in scale 0...1
    def predict_score(self, test, correct_result):
        pred = self.predict(test)
        return np.mean(pred == correct_result)
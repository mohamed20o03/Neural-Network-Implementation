import numpy as np
from Value import Value

class Neuron:

    def __init__(self, N_input):
        self.W = list(map(Value, np.random.uniform(-1, 1, (N_input))))
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, X_input):
        out = sum([Wi * Xi for Wi, Xi in zip(self.W, X_input)]) + self.b.data
        return out.tanh()

    @property
    def parameters(self):
        return self.W + [self.b]

class layer:

    def __init__(self, N_input, N_neuron):
        self.neurons = [Neuron(N_input) for _ in range(N_neuron)]

    def __call__(self, X):
        out = [neuron(X) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    @property
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters]


class MLP:

    def __init__(self, N_input, layers_siz):
        self.layer_input = [N_input] + layers_siz
        self.layers = [layer(self.layer_input[i], self.layer_input[i + 1]) for i in range(len(self.layer_input) - 1)]

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]
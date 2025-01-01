import numpy as np

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - z ** 2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

class XORNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = tanh(self.z1)
        
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        # Error at the hidden layer
        hidden_error = np.dot(output_delta, self.w2.T)
        hidden_delta = hidden_error * tanh_derivative(self.a1)
        
        # Update weights and biases
        self.w2 += np.dot(self.a1.T, output_delta) * learning_rate
        self.b2 += np.sum(output_delta, axis=0) * learning_rate
        
        self.w1 += np.dot(X.T, hidden_delta) * learning_rate
        self.b1 += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs, aplha):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, aplha)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1
nn = XORNeuralNetwork(input_size, hidden_size, output_size)

epochs = 10000
learning_rate = 0.2
nn.train(X, y, epochs, learning_rate)

print("Trained XOR Neural Network Results:")
for i in range(len(X)):
    output = nn.forward(X[i])
    prediction = 1 if output >= 0.5 else 0
    print(f"Input: {X[i]} -> Predicted Output: {prediction}")

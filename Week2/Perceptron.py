import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size) * 0.1
        self.bias = np.random.rand(1) * 0.1
        self.alpha = 0.1  

    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(z)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(X)):
                z = np.dot(self.weights, X[i]) + self.bias
                prediction = self.activation_function(z)
                gradient = (y[i] - prediction) * prediction * (1 - prediction)
                self.weights += self.alpha * gradient * X[i]
                self.bias += self.alpha * gradient

def train_and_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    and_gate = Perceptron(input_size=2)
    and_gate.train(X, y, epochs=5000)
    return and_gate


and_gate_per = train_and_gate()

def and_gate(A,B):
    return 1 if and_gate_per.predict(np.array([A,B]))>=0.5 else 0

# # ====================AND OUTPUT :==============================
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# for inputs in X:
#     print("Input:", inputs, "-> AND Output:", and_gate(inputs[0],inputs[1]))

def train_or_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    or_gate = Perceptron(input_size=2)
    or_gate.train(X, y, epochs=5000)
    return or_gate

def not_gate(input_value):
    return 1 if input_value < 0.5 else 0  # Simple NOT logic

or_gate_per = train_or_gate()

def or_gate(A,B):
    return 1 if or_gate_per.predict(np.array([A,B]))>=0.5 else 0

def xor_gate(A, B):
    # A AND NOT B
    not_B = not_gate(B)
    and1 = and_gate(A,not_B)

    # NOT A AND B
    not_A = not_gate(A)
    and2 = and_gate(not_A,B)

    # Combine with OR
    xor_output = or_gate(and1,and2)
    return 1 if xor_output >= 0.5 else 0

# # ====================XOR OUTPUT :==============================
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# for inputs in X:
#     print("Input:", inputs, "-> XOR Output:", xor_gate(inputs[0], inputs[1]))

def FullAdder (A,B,CIN):
    xor1=xor_gate(A,B)
    and1=and_gate(A,B)
    and2=and_gate(xor1,CIN)

    Sum = xor_gate(xor1,CIN) 
    C_OUT = or_gate(and1,and2)

    return Sum,C_OUT

# # ==============FULL ADDER========================================
# inputs = [
#     (0, 0, 0),
#     (0, 0, 1),
#     (0, 1, 0),
#     (0, 1, 1),
#     (1, 0, 0),
#     (1, 0, 1),
#     (1, 1, 0),
#     (1, 1, 1),
# ]

# print("Full Adder Results:")
# for A, B, C_IN in inputs:
#     Sum, C_OUT = FullAdder(A, B, C_IN)
#     print(f"Input: A={A}, B={B}, C_IN={C_IN} -> Sum={Sum}, C_OUT={C_OUT}")

def RippleCarryAdder(A,B):
    binA=list(bin(A)[2:])
    binB=list(bin(B)[2:])
    
    if len(binA)>len(binB):
        binB = ['0']*(len(binA)-len(binB)) + binB

    elif len(binA)<len(binB):
        binA = ['0']*(len(binB)-len(binA)) + binA

    carry=0
    result = []

    binA_rev=binA[::-1]
    binB_rev=binB[::-1]

    for i in range(len(binA)):
        Sum,carry= FullAdder(int(binA_rev[i]),int(binB_rev[i]),carry)
        result.append(str(Sum))

    result.append(str(carry))
    result=result[::-1]
    result="".join(result)
    result=int(result,2)
    return result

a=int(input("Enter number 1 to add : "))
b=int(input("Enter number 2 to add : "))

print("The sum is : ", RippleCarryAdder(a,b))


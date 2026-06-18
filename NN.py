# Input X is assumed to be numpy always

import numpy as np


rng = np.random.default_rng()


X = np.array([5.1, 3.5, 1.4, 0.2])
Ground_Truth = np.array([1.0, 0.0])
Learning_Rate = 1e-4

alpha = 1.6732632423543772848170429916717
lambda_val = 1.0507009873554804934193349852946
def activation_function(X):
    Y = []
    for k in X:
        if k > 0:
            Y.append(lambda_val)
        else:
            Y.append(alpha * lambda_val * np.exp(k))
    Y = np.array(Y)
    return Y

class NNLayer():
    def __init__(self, inp_dim, out_dim):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.weights =  1200 * rng.random((inp_dim,out_dim))
        self.bias = rng.random(out_dim)
        self.Gradient = 0
        self.derivative = 0

    def forward(self, X):
        Dot_Product = (self.weights).T @ X 
        self.layer_sum = Dot_Product + self.bias
        output = activation_function(self.layer_sum)
        return output

    
    def update(self,loss_matrix,X):
        Gradient = np.outer(X , loss_matrix)
        self.derivative = Gradient
        self.weights = self.weights - Gradient*Learning_Rate
        self.bias = self.bias - loss_matrix * Learning_Rate
    def convert_to_probaility(self,output):
        prb = []
        for o in output:
            prb.append(1/(1 + np.exp(-1 * o)))
        return prb
    
    def backpropagate(self, loss_matrix, layer_output):
        derivatives = []
        for k in self.layer_sum:
            if k > 0:
                derivatives.append(lambda_val)
            else:
                derivatives.append(alpha * lambda_val * np.exp(k))
        derivatives = np.array(derivatives)
        activations = loss_matrix * derivatives
        product = self.weights @ activations
        return product

def calculate_loss(Values, Ground_Truth):

    Loss  = 0
    for x_p,x_r in zip(Values, Ground_Truth):
        Loss += pow((x_p - x_r),2)
    Loss = Loss/len(Ground_Truth)
    Loss = pow(Loss,0.5)        
    #Loss = Loss/len(Ground_Truth)
    return Loss


    

    

layer1 = NNLayer(inp_dim=4,out_dim=5)
layer2 = NNLayer(inp_dim=5,out_dim=10)
layer3 = NNLayer(inp_dim=10,out_dim=100)
layer4 = NNLayer(inp_dim=100,out_dim=2)


n_epochs = 1000
for n in range(n_epochs):
    output1 = layer1.forward(X)
    output2 = layer2.forward(output1)
    output3 = layer3.forward(output2)
    output4 = layer4.forward(output3)
    
    Errors = output4 - Ground_Truth


    print(f"Loss @ {n} :",calculate_loss(Errors,Ground_Truth))
    hidden_error4 = layer4.backpropagate(Errors,output4)
    layer4.update(Errors,output3)
    hidden_errors3 = layer3.backpropagate(hidden_error4, output3)
    layer3.update(hidden_error4,output2)

    hidden_errors2 = layer2.backpropagate(hidden_errors3, output2)
    layer2.update(hidden_errors3,output1)
    hidden_errors1 = layer1.backpropagate(hidden_errors2, output1)
    layer1.update(hidden_errors2,X)



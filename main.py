import numpy as np
from utils import *

'''class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output'''

def getRandomMatrix(*dimensions):
    return np.random.random_sample(dimensions)

def RELU(matrix):
    print("executing RELU")
    new_matrix = []
    for i in np.nditer(matrix):
        if i < 0:
            new_matrix.append(0)
        else:
            new_matrix.append(i)
    new_matrix = np.array(new_matrix)
    return new_matrix.reshape(matrix.shape)

def sigmoid(matrix):
    print("executing sigmoid")
    return (1.0/(1+np.exp(-matrix)))

def tanh(matrix):
    print("executing tanh")
    return (np.exp(matrix)-np.exp(-matrix))/(np.exp(matrix)+np.exp(-matrix))


def SoftMax():
    return 


class MLP:
    def __init__(self, hidden_num, layer_functs, node_num):
        # (self, int, [funciones], [ints])
        print("me construyo")
        assert(hidden_num == len(layer_functs))
        assert(hidden_num == len(node_num)-1) #notar que no deberiamos contar la capa de input porque no hay una funcion pasando por ahi
        self.input_size = node_num[0]
        self.layers = layer_functs
        self.w = []
        for i in range(1,len(node_num)):
            self.w.append(getRandomMatrix(node_num[i],node_num[i-1]))
        for ws in self.w:
            print("-",ws)         

    
    #w {#neuronas capa 1, X} * input {X, 1}   = ans {#neuronas, 1} => layers[i](ans)

    def forward(self, item):
        temp_ans = np.array([item])
        for i in range(len(self.layers)):
            print("Layer", i+1)
            print("arr shape    :", temp_ans.shape)
            print("w   shape    :", self.w[i].T.shape)
            temp_ans = np.dot(temp_ans,self.w[i].T)
            print("arr after    :", temp_ans.shape)
            temp_ans = self.layers[i](temp_ans)
            print("arr activated:", temp_ans.shape)
            print("----------")
        return temp_ans
    
    def backPropagate(self,output,expected_output):
        xd = 0
        
    def execute(self, dataset):
        counter = 1
        for item in dataset:
            print("COUNTER #",counter)
            output = self.forward(np.array(item[0]).astype("float"))
            self.backPropagate(output,item[1])
            return
            
            

if __name__ == '__main__':
    mlp = MLP(4,[RELU,sigmoid,RELU,tanh],[30,20,40,20,10])
    dataset = read_db()
    mlp.execute(dataset)    
    # def forward():

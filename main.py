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

def getRandomMatrix(*dimensions): # Pero para que necesitas esto si dimensions en primer lugar ya es una lista XD
    return np.random.random_sample(dimensions) #todo bien manito? XD

def RELU(matrix):
    new_matrix = []
    for i in np.nditer(matrix):
        if i < 0:
            new_matrix.append(0)
        else:
            new_matrix.append(i)
    new_matrix = np.array(new_matrix)
    return new_matrix.reshape(matrix.shape)

def SoftMax():
    return 


# Sigmoid, RELU, 
class MLP:
    def __init__(self, hidden_num, layer_functs, node_num):
        # (self, int, [funciones], [ints])
        print("me construyo")
        assert(hidden_num == len(layer_functs))
        assert(hidden_num == len(node_num))
        self.input_size = node_num[0]
        self.layers = layer_functs
        self.w = [getRandomMatrix(1,i) for i in node_num]
        for ws in self.w:
            print(ws)         

    
    #w {#neuronas capa 1, X} * input {X, 1}   = ans {#neuronas, 1} => layers[i](ans)

    def forward(self, item):
        temp_ans = item
        for i in range(len(self.layers)):
            temp_ans = self.w[i] * temp_ans.T
            temp_ans = self.layers[i](temp_ans)
        return temp_ans
    
    def backPropagate(self,output,expected_output):
        xd = 0 
        
    def execute(self, dataset):
        for item in dataset:
            output = self.forward(np.array(item[0]).astype("float"))
            self.backPropagate(output,item[1])
            
            

if __name__ == '__main__':
    mlp = MLP(3,[RELU,'sigmoid','tanh'],[30,20,10])
    dataset = read_db()
    mlp.execute(dataset)    
    # def forward():

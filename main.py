from scipy import stats
import numpy as np
from utils import *
import decimal
import random
from sklearn.model_selection import train_test_split

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
    return np.random.uniform(0,1,dimensions)

def RELU(matrix):
    #print("executing RELU")
    new_matrix = []
    for i in np.nditer(matrix):
        if i < 0:
            new_matrix.append(0)
        else:
            new_matrix.append(i)
    new_matrix = np.array(new_matrix)
    return new_matrix.reshape(matrix.shape)

def devRELU(matrix):
    #print("executing RELU")
    new_matrix = []
    for i in np.nditer(matrix):
        if i < 0:
            new_matrix.append(0)
        else:
            new_matrix.append(1)
    new_matrix = np.array(new_matrix)
    return new_matrix.reshape(matrix.shape)

class Sigmoid:
    def calculate(matrix):
        #print("executing sigmoid")
        return (1.0/(1+np.exp(-matrix)))
    def derivative(vec):
        #print("executing sigmoid derivative")
        return vec*(1-vec)  

class Tanh:
    def calculate(matrix):
        #print("executing tanh")
        return (np.exp(matrix)-np.exp(-matrix))/(np.exp(matrix)+np.exp(-matrix))

    def derivative(matrix):
        #print("executing tanh derivative")
        return (4*np.exp(2*matrix))/np.power((np.exp(2*matrix)+1), 2)


def tanh(matrix):
    #print("executing tanh")
    return (np.exp(matrix)-np.exp(-matrix))/(np.exp(matrix)+np.exp(-matrix))

def devTanh(matrix):
    #print("executing tanh derivative")
    return (4*np.exp(2*matrix))/np.power((np.exp(2*matrix)+1), 2)

def softMax(vect):
    return np.exp(vect)/sum(np.exp(vect))

    
class MLP:
    def __init__(self, hidden_num, layer_functs, node_num):
        # (self, int, [funciones], [ints])
        print("me construyo")
        assert(hidden_num == len(layer_functs))
        assert(hidden_num == len(node_num)-1) # notar que no deberiamos contar la capa de input porque no hay una funcion pasando por ahi
        self.input_size = node_num[0]
        self.layers = layer_functs
        self.w = []
        self.delta = []
        self.S = []
        for i in range(1,len(node_num)):
            self.delta.append(np.zeros(node_num[i])) # K layers X nodos Y ->cantidad de nodos del siguiente layer
            #self.S.append(np.zeros(node_num[i])) #(np.zeros((node_num[i-1],node_num[i]))) # K layers X nodos Y ->cantidad de nodos del siguiente layer
            self.w.append(getRandomMatrix(node_num[i],node_num[i-1]))
        #for ws in self.w:
        #    print("-",ws)


    #w {#neuronas capa 1, X} * input {X, 1}   = ans {#neuronas, 1} => layers[i](ans)

    def forward(self, item):
        #print(item)
        self.S = []
        self.S.append(np.array(item).reshape(len(item),1))
        temp_ans = np.array([item])
        for i in range(len(self.layers)):
            #print("Layer", i+1)
            #print("arr shape    :", temp_ans.shape)
            #print("w   shape    :", self.w[i].T.shape)
            temp_ans = np.dot(temp_ans,self.w[i].T)
            #print("temp_ans = ",temp_ans)
            #print("arr after    :", temp_ans.shape)
            temp_ans = self.layers[i].calculate(temp_ans)
            #print("arr activated:", temp_ans.shape)
            #print("----------")
            #print("temp_ans = ",temp_ans)
            self.S.append(temp_ans.T)
        return temp_ans
        
    def backPropagate(self,expected_output,alpha):
        print(len(self.S))
        print(len(self.w))
        print(len(self.layers))
        for i in range(len(self.w),1,-1):
            print(i)
            if i == len(self.w)-1:
                #print(self.S[i].shape)
                #print("#######")
                #print(np.array(self.layers[i-1].derivative(self.S[i])).shape)
                #print("#######")
                #print("expected_output is",expected_output.shape)
                #print("#######")
                first_mult = (expected_output- self.S[i])*(-1) * np.array(self.layers[i-1].derivative(self.S[i]))
                #print(first_mult.shape)
                #print("#######")
                #print(self.S[i-1].shape)
                #print("#######")
                dev = np.dot(self.S[i-1],first_mult.T)
                self.delta[i-1] = first_mult.T
                self.w[i-1] = self.w[i-1] - (alpha*dev).T
            else:
                #print("------")
                #print(self.delta[i].shape)
                #print(self.w[i].shape)
                #print(np.dot(self.delta[i],self.w[i]).shape)
                #print(self.S[i].shape)
                #print("------")
                #print(np.array(self.layers[i-1].derivative(self.S[i])).shape)
                print(self.delta[i])
                first_mult = np.dot(self.delta[i],self.w[i-1]) * np.array(self.layers[i-1].derivative(self.S[i])).T
                #print("------")
                #print(first_mult.shape) 
                dev = np.dot(self.S[i-1],first_mult)
                self.delta[i-1] = first_mult
                self.w[i-1] = self.w[i-1] - (alpha*dev).T

    def dist(self,a,b): 
        return np.linalg.norm(a-b)
        
    def execute(self, dataset,iters =20000):
        counter = 1
        for i in range(iters):
            item = dataset[random.randint(0,len(dataset)-1)]
        #for item in dataset:
            #print("COUNTER #",counter)
            output = self.forward(stats.zscore(np.array(item[0]).astype("float")))
            #print("output", output)
            #print("self.S", self.S)
            oneHotEncoder = np.zeros(2)
            oneHotEncoder[item[1]]=1
            oneHotEncoder = oneHotEncoder.reshape(2,1)
            print(oneHotEncoder)
            print(output)
            print(softMax(output))
            print("error is:",self.dist(softMax(output),oneHotEncoder))
            self.backPropagate(oneHotEncoder,0.007)
        return

    def predict(self, dataset): 
        goods = 0
        bads = 0
        for item in dataset:
            #for item in dataset:
            output = self.forward(stats.zscore(np.array(item[0]).astype("float")))
            #print("output", output)
            #print("self.S", self.S)
            #print(np.argmax(output))
            #print(item[1])
            
            if (np.argmax(output)==item[1]):
                goods +=1
            else:
                bads+=1
        print("there are ",goods," goods")
        print("there are ",bads," bads")
        print("accuracy is: ",goods/(goods+bads))
        return 
            
            

if __name__ == '__main__':
    mlp = MLP(2,[Tanh,Tanh],[30,15,2])
    dataset = read_db()
    train,test = train_test_split(dataset,test_size = 0.3,shuffle=True)
    mlp.execute(train)    
    mlp.predict(test)
    # def forward():

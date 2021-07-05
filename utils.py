
import csv
import numpy as np 
from scipy import stats


types={
    "M":1,
    "B":0
}

def read_db():
    reader = csv.reader(open("cancer.csv", "r"), delimiter=",")
    data = list(reader)
    dataset = [(line[2:],types[line[1]]) for line in data]
    Y = [line[1] for line in data]
    #print(dataset)
    #result = numpy.array(X).astype("float")
    return dataset 


def read_db2():
    reader = csv.reader(open("cancer.csv", "r"), delimiter=",")
    data = list(reader)
    dataset = [ list(np.float_(line[2:])) for line in data]
    Y = [types[line[1]] for line in data]
    #print(dataset)
    #result = numpy.array(X).astype("float")
    return stats.zscore(dataset), Y 

#a = np.array([[1],[2],[3]])
#b = np.array([[1,2,3]])
#print(np.dot(a,b))
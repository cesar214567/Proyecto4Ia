
import csv
import numpy as np 

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


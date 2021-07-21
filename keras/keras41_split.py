import numpy as np

a = np.array(range(1, 11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

print(dataset) 

'''
[[ 1  2  3  4  5] 
 [ 2  3  4  5  6] 
 [ 3  4  5  6  7] 
 [ 4  5  6  7  8] 
 [ 5  6  7  8  9] 
 [ 6  7  8  9 10]]
'''

x = dataset[:, :4]
y = dataset[:, 4]

print("x : ", x) 
print("y : ", y) 
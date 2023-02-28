import numpy as np
'''
a = np.arange(45).reshape(5,3,3)
print("a =")
print(a)
b = a.reshape(5,a.shape[1]*a.shape[2])
print("b =")
print(b)
c = b.reshape(5,3,3)
print("c =")
print(c)
'''

d = np.array([[2], [4], [7], [4], [0], [5], [9]])
print("d = ")
print(d)
e = np.zeros(shape = (d.shape[0],10))

for i in range(d.shape[0]):
    e[i,d[i,0]] = 1
    #print(d[i,0])
#print("modified d:")
#print(d)
print("e = ")
print(e)


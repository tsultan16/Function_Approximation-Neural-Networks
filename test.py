import numpy as np
from scipy.special import logsumexp


a = np.array([[1,1,6], [4,4,2], [6, 0, 2]])
print(a)
S = logsumexp(a, axis = 1)
print(S)
print(np.log(np.sum(np.exp(a), axis=1)))
a_exp = np.reshape(S, (3,1))
print(a_exp)
#print(a-a_exp)
sm = np.exp(a-a_exp)
print(sm)
'''
b = 1/np.sum(a, axis = 1)
c = np.reshape(b, (3,1))
print(b)
print(c)
print(a*c)
'''
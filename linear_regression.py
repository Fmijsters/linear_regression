import numpy as np
import matplotlib.pyplot as plt
number_of_parameters = 13
np.set_printoptions(suppress=True) 
data = np.loadtxt("ajax1516.txt",dtype=np.float64,delimiter=",")
X = data[::,0:number_of_parameters-1]
Y = data[::,-1:]

Theta = np.random.rand(1,number_of_parameters)
m,n = X.shape

X_bias = np.ones((m,n+1))
X_bias[::,1:] = X

mean_array = {}
std_array= {}
for x in range(2,number_of_parameters+1):
	mean_array[x-2] = np.mean(X_bias[::,x-1:x])
	std_array[x-2] = np.std(X_bias[::,x-1:x])

for x in range(2,number_of_parameters+1):
	array_index = x - 2
	X_bias[::,x-1:x] = (X_bias[::,x-1:x] - mean_array[array_index]) / (std_array[array_index])

def cost(X_bias,Y,Theta):
    np.seterr(over='raise')
    m,n = X.shape
    hypothesis = X_bias.dot(Theta.transpose())
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))

def gradientDescent(X_bias,Y,Theta,iterations,alpha):
    count = 1
    cost_log = np.array([])

    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())
        temp_array={}
        for x in range(1,number_of_parameters+1):
        	temp_array[x-1] = Theta[0,x-1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,x-1:x])).sum(axis=0)
        	Theta[0,x-1] = temp_array[x-1]

        cost_log = np.append(cost_log,cost(X_bias,Y,Theta))
        count = count + 1
    return Theta

alpha = 0.3
iterations = 100

total =0.0
for i in range(2):
	Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)
	X_predict = np.array([1.0,180,65,35,3043,9,15,4,0,4.9,75.4,0.2,9]) 
	
	for x in range(1,number_of_parameters):
		X_predict[x] = (X_predict[x] - mean_array[x-1])/ (std_array[x-1]) 
	
	hypothesis = X_predict.dot(Theta.transpose())
	# print(hypothesis[0])
	total = total + hypothesis[0]
print("player on line 26 we think is ",total/2)

import numpy as np
import matplotlib.pyplot as plt
# number_of_parameters = 13
number_of_parameters = 26
np.set_printoptions(suppress=True) 
# filename = "ajax1516.txt"
filename = "ajax1718.txt"
data = np.loadtxt(filename,dtype=np.float64,delimiter=",")
X = data[::,0:number_of_parameters-1]
Y = data[::,-1:]

Theta = np.random.rand(1,number_of_parameters)
m,n = X.shape
print(m,n)
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
    # print(hypothesis)
    # print(Y)
    # print(hypothesis-Y)
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))

def gradientDescent(X_bias,Y,Theta,iterations,alpha):
    count = 1
    cost_log = np.array([])

    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())
        temp_array={}
        for x in range(1,number_of_parameters):
        	temp_array[x-1] = Theta[0,x-1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,x-1:x])).sum(axis=0)
        for x in range(1,number_of_parameters):
        	Theta[0,x-1] = temp_array[x-1]
        cost_log = np.append(cost_log,cost(X_bias,Y,Theta))
        count = count + 1
    plt.plot(np.linspace(1,iterations,iterations,endpoint=True),cost_log)
    plt.title("Iteration vs Cost graph ")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost of Theta")
    plt.show()
    return Theta

total = 0.0

for x in range(1):
	
	alpha = 0.3
	iterations = 100
	Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)
	# print(Theta)
	X_predict = np.array([1.0,180,65,6,525,2.3,0.8,1.5,0,0,0.8,0,2,2,5.2,3.2,2.5,2,0.2,2.8,3.7,55.3,78.3,2,4,1])
	# X_predict = np.array([1.0,179,65,21,1791,1,5,3,0,0.5,84.7,1.1,1]) 
	for x in range(1,number_of_parameters):
		X_predict[x] = (X_predict[x] - mean_array[x-1])/ (std_array[x-1]) 
	hypothesis = X_predict.dot(Theta.transpose())
	# print(hypothesis[0])
	total = total + hypothesis[0]
print("player on line 26 we think is ",total/1)

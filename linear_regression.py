import numpy as np
import matplotlib.pyplot as plt
import csv
# number_of_parameters = 13
number_of_parameters = 30
# number_of_parameters = 5
np.set_printoptions(suppress=True) 
# filename = "data/ajax1516.txt"
# filename = "data/ajax1718.txt"
filename = "data/wholeAjaxEredivisie.txt"
# filename = "data/netflixdata.txt"
data = np.loadtxt(filename,dtype=np.float64,delimiter=",")
X = data[::,0:number_of_parameters-1]
Y = data[::,-1:]
print(Y)
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
def predict(X_predict):
	for x in range(1,number_of_parameters):
		X_predict[x] = (X_predict[x] - mean_array[x-1])/ (std_array[x-1]) 
	hypothesis = X_predict.dot(Theta.transpose())
	print("We think the value for the specified data is: " + str(hypothesis[0]))


for x in range(1):
	
	alpha = 0.01
	iterations = 100000
	Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)
	# print(Theta)
	# X_predict = np.array([1.0,180,65,6,525,2.3,0.8,1.5,0,0,0.8,0,2,2,5.2,3.2,2.5,2,0.2,2.8,3.7,55.3,78.3,2,4,1])
	X_predict = np.array([1.0,177,78,27,2350,10,3,6,0,1.9,87,1.9,2,2.1,1.3,0.9,0,1,1.1,0.2,0,1.4,1,0,1,1.2,56.1,0.9,1.7,0.1])
	# X_predict = np.array([1.0,179,65,21,1791,1,5,3,0,0.5,84.7,1.1,1]) 

	# with open('data/needpredicting.txt') as csvfile:
	# 	smartreader = csv.DictReader(csvfile)
	# 	f = open("data/predictedmovies.txt", "w")
	# 	teller = 1
	# 	for row in smartreader:
	# 		if row["number"] is None or row["sex"]is None or ["proffesion"]is None or row["age"] is None or row["release"] is None:
	# 			print(str(teller) + " this is none")
	# 			continue
	# 		X_predict = np.array([int(row["number"]),int(row["sex"]),int(row["proffesion"]),int(row["age"]),int(row["release"])]) 
	# 		# print(row)
	# X_predict = np.array([int(row["number"]),int(row["sex"]),int(row["proffesion"]),int(row["age"]),int(row["release"])]) 
	predict(X_predict)
	# print(hypothesis[0])
			# f.write("movie on line "+str(teller)+" we think is " + str(hypothesis[0]) + "\n")
			# teller = teller +1

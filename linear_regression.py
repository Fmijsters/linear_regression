import numpy as np
import matplotlib.pyplot as plt
#supressing the scientific output
np.set_printoptions(suppress=True) 
data = np.loadtxt("statsdefajax.txt",dtype=np.float64,delimiter=",")
# print(data)
X = data[::,0:12]
Y = data[::,-1:]
# print(X)
plt.figure(figsize = (15,4),dpi=100)
plt.subplot(121)
plt.scatter(X[::,0:1],Y)
plt.xlabel("CM (X1)")
plt.ylabel("Rating (Y)")
plt.subplot(122)
plt.scatter(X[::,1:2],Y)
plt.xlabel("KG (X2)")
plt.ylabel("Rating (Y)")
# plt.show()

# introduce weights of hypothesis (randomly initialize)
Theta = np.random.rand(1,13)
# m is total example set , n is number of features
m,n = X.shape
# print(n)
# add bias to input matrix by simple make X0 = 1 for all
X_bias = np.ones((m,n+1))
X_bias[::,1:] = X
# output first 5 X_bias examples

#feature scaling
# it also protect program from overflow error
#Gemiddelde van m2
mean_cm = np.mean(X_bias[::,1:2])

mean_KG = np.mean(X_bias[::,2:3])
mean_Apps = np.mean(X_bias[::,3:4])
mean_Mins = np.mean(X_bias[::,4:5])
mean_Goals = np.mean(X_bias[::,5:6])
mean_Assists = np.mean(X_bias[::,6:7])
mean_Yel = np.mean(X_bias[::,7:8])
mean_Red = np.mean(X_bias[::,8:9])
mean_SpG = np.mean(X_bias[::,9:10])
mean_PS = np.mean(X_bias[::,10:11])
mean_AerialsWon = np.mean(X_bias[::,11:12])
mean_MotM = np.mean(X_bias[::,12:13])
#standard deviation gemiddelde verschil met het gemiddelde gekwadrateerd
std_cm = np.std(X_bias[::,1:2])

std_KG = np.std(X_bias[::,2:3])
std_Apps = np.std(X_bias[::,3:4])
std_Mins = np.std(X_bias[::,4:5])
std_Goals = np.std(X_bias[::,5:6])
std_Assists = np.std(X_bias[::,6:7])
std_Yel = np.std(X_bias[::,7:8])
std_Red = np.std(X_bias[::,8:9])
std_SpG = np.std(X_bias[::,9:10])
std_PS = np.std(X_bias[::,10:11])
std_AerialsWon = np.std(X_bias[::,11:12])
std_MotM = np.std(X_bias[::,12:13])


# print(X_bias[::,1:2])
X_bias[::,1:2] = (X_bias[::,1:2] - mean_cm)/ (std_cm) 
# print(X_bias[::,1:2])
X_bias[::,2:3] = (X_bias[::,2:3] - mean_KG)/ (std_KG) 
X_bias[::,3:4] = (X_bias[::,3:4] - mean_Apps)/ (std_Apps) 
X_bias[::,4:5] = (X_bias[::,4:5] - mean_Mins)/ (std_Mins) 
X_bias[::,5:6] = (X_bias[::,5:6] - mean_Goals)/ (std_Goals) 
X_bias[::,6:7] = (X_bias[::,6:7] - mean_Assists)/ (std_Assists) 
X_bias[::,7:8] = (X_bias[::,7:8] - mean_Yel)/ (std_Yel) 
X_bias[::,8:9] = (X_bias[::,8:9] - mean_Red)/ (std_Red) 
X_bias[::,9:10] = (X_bias[::,9:10] - mean_SpG)/ (std_SpG) 
X_bias[::,10:11] = (X_bias[::,10:11] - mean_PS)/ (std_PS) 
X_bias[::,11:12] = (X_bias[::,11:12] - mean_AerialsWon)/ (std_AerialsWon) 
X_bias[::,12:13] = (X_bias[::,12:13] - mean_MotM)/ (std_MotM) 


#haal van alle m2 data de gemiddelde grootte af en deel het dan door de standard deviation
# X_bias[::,1:2] = (X_bias[::,1:2] - mean_size)/ (size_std) 
#zelfde voor de slaap kamers
# X_bias[::,2:] = (X_bias[::,2:] - mean_bedroom)/ (bedroom_std)

#maak de tabel weer heel met de nieuwe getallen
# print(X_bias)
# X_bias[0:5,::]
# print(X_bias)



def cost(X_bias,Y,Theta):
    np.seterr(over='raise')
    m,n = X.shape
    hypothesis = X_bias.dot(Theta.transpose())
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))

def gradientDescent(X_bias,Y,Theta,iterations,alpha):
	#prices
    # print("Y %s",Y)
    #random number between 0.1 and 0.3
    # print("Theta %s",Theta)
    #100 standard
    # print(iterations)
    # print("Alpha %s",alpha)
    count = 1

    cost_log = np.array([])

    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())

        temp0 = Theta[0,0] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,0:1])).sum(axis=0)
        temp1 = Theta[0,1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,1:2])).sum(axis=0)
        temp2 = Theta[0,2] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,2:3])).sum(axis=0)
        temp3 = Theta[0,3] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,3:4])).sum(axis=0)
        temp4 = Theta[0,4] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,4:5])).sum(axis=0)
        temp5 = Theta[0,5] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,5:6])).sum(axis=0)
        temp6 = Theta[0,6] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,6:7])).sum(axis=0)
        temp7 = Theta[0,7] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,7:8])).sum(axis=0)
        temp8 = Theta[0,8] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,8:9])).sum(axis=0)
        temp9 = Theta[0,9] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,9:10])).sum(axis=0)
        temp10 = Theta[0,10] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,10:11])).sum(axis=0)
        temp11 = Theta[0,11] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,11:12])).sum(axis=0)
        temp12 = Theta[0,12] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,12:13])).sum(axis=0)
        # temp13 = Theta[0,13] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,-1:])).sum(axis=0)
        Theta[0,0] = temp0
        Theta[0,1] = temp1
        Theta[0,2] = temp2
        Theta[0,3] = temp3
        Theta[0,4] = temp4
        Theta[0,5] = temp5
        Theta[0,6] = temp6
        Theta[0,7] = temp7
        Theta[0,8] = temp8
        Theta[0,9] = temp9
        Theta[0,10] = temp10
        Theta[0,11] = temp11
        Theta[0,12] = temp12
        # Theta[0,13] = temp13


        cost_log = np.append(cost_log,cost(X_bias,Y,Theta))
        count = count + 1
    # print(Theta)
    plt.plot(np.linspace(1,iterations,iterations,endpoint=True),cost_log)
    plt.title("Iteration vs Cost graph ")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost of Theta")
    # plt.show()
    return Theta

alpha = 0.3
iterations = 100






total =0.0
for i in range(100):
	Theta = gradientDescent(X_bias,Y,Theta,iterations,alpha)
	X_predict = np.array([1.0,180,65,35,3043,9,15,4,0,4.9,75.4,0.2,9]) 
	# print(len(X_predict))
	#feature scaling the data first
	X_predict[1] = (X_predict[1] - mean_cm)/ (std_cm) 
	X_predict[2] = (X_predict[2]- mean_KG)/ (std_KG)
	X_predict[3] = (X_predict[3]- mean_Apps)/ (std_Apps)
	X_predict[4] = (X_predict[4]- mean_Mins)/ (std_Mins)
	X_predict[5] = (X_predict[5]- mean_Goals)/ (std_Goals)
	X_predict[6] = (X_predict[6]- mean_Assists)/ (std_Assists)
	X_predict[7] = (X_predict[7]- mean_Yel)/ (std_Yel)
	X_predict[8] = (X_predict[8]- mean_Red)/ (std_Red)
	X_predict[9] = (X_predict[9]- mean_SpG)/ (std_SpG)
	X_predict[10] = (X_predict[10]- mean_PS)/ (std_PS)
	X_predict[11] = (X_predict[11]- mean_AerialsWon)/ (std_AerialsWon)
	X_predict[12] = (X_predict[12]- mean_MotM)/ (std_MotM)
	hypothesis = X_predict.dot(Theta.transpose())
	# print(hypothesis[0])
	total = total + hypothesis[0]
print("player on line 26 we think is ",total/100)

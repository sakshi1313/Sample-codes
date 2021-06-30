import numpy as np
#m=4 n=2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #inputs
y=np.array([0,1,1,0]) #output
y=y.reshape(4,1)
#print(X.shape) 2X4

m=X.shape[0]
#print(m)
#no. of inputs=2
#no. of neurons in hidden layer=2
alpha=0.05
bias1=0.2
bias2=0.2
np.random.seed(1)

Theta1=2*np.random.rand(2,2) -1 #2x2
Theta2=2*np.random.rand(1,2) -1 #1x2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def for_prop(Theta1,Theta2,X):
    z1=np.dot(X,Theta1) +bias1 #4x2
    a1=sigmoid(z1)
    z2=np.dot(a1,Theta2.T) +bias2#4x1
    a2=sigmoid(z2)
    h=a2 #output #4x1
    #print(h.shape)
    return z1,a1,z2,h

def back_prop(Theta1,Theta2,z1,z2,a1,h,y):
    del2=h-y #4x1
    del1=np.dot(del2,Theta2)*sigmoid(z1) #4x2

    der_Theta2=np.dot(del2.T,a1)/m #1X2
    der_Theta1=np.dot(del1.T,X)/m#2x2
    der_bias2=np.sum(del2)/m
    der_bias1=(np.sum(del1))/m



    return del2,del1,der_Theta1,der_Theta2,der_bias1,der_bias2

#z1,a1,z2,h=for_prop(Theta1,Theta2,X)
#back_prop(Theta1,Theta2,z1,z2,a1,h,y)

for i in range(10000):
    z1, a1, z2, h = for_prop(Theta1, Theta2, X)
    error_func = y * np.log(h) + (1 - y) * np.log(1 - h)
    J = -(1/m)*(np.sum(error_func))
    #print(J)
    del2, del1, der_Theta1, der_Theta2,der_bias1,der_bias2=back_prop(Theta1,Theta2,z1,z2,a1,h,y)

    Theta2=Theta2-alpha*der_Theta2
    Theta1=Theta1-alpha*der_Theta1
    bias1=bias1-der_bias1
    bias2 = bias2 - der_bias2


np.set_printoptions(suppress=True)
print("Predicted values:")
print(h.T) # 'h' is the predicted output





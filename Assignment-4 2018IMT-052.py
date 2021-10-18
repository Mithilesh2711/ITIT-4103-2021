# -*- coding: utf-8 -*-
"""2018IMT-052_Assignment_4


Name:        Mithilesh Kumar
Roll no.:    2018IMT-052
Course:      Machine Learning
Course Code: ITIT 4107-2021
Assignment-4
Deadline: 18th Oct 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("https://raw.githubusercontent.com/AlfTang/Linear-Regression/master/ex1data1.txt",names = ['population','profit'],header = None)

df.head()

df.describe()


plt.scatter(df['population'],df['profit'])
plt.title("Profit Vs Population")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population")
plt.ylabel("Profit")

def computeCost(X,y,theta):
    m=len(y)
    predictions=X.dot(theta)
    square_err=(predictions - y)**2
    
    return 1/(2*m) * np.sum(square_err)

data_n=df.values
m=len(data_n[:,-1])
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
    
    m=len(y)
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(),(predictions -y))
        descent=alpha * 1/m * error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    
    return theta, J_history

def plot_model_fit(alpha,theta):
    plt.scatter(df['population'],df['profit'])
    plt.title("Profit vs Population with alpha = {}\n".format(alpha))
    x_value=[x for x in range(25)]
    y_value=[y*theta[1]+theta[0] for y in x_value]
    plt.plot(x_value,y_value,color="r")
    plt.xticks(np.arange(5,30,step=5))
    plt.yticks(np.arange(-5,30,step=5))
    plt.xlabel("Population")
    plt.ylabel("Profit")

def func(i):
  theta=np.zeros((2,1))
  alpha = alpha_values[i]
  theta,J_history = gradientDescent(X,y,theta,alpha,100)
  print('The ALPHA value, learning_rate: {}'.format(alpha))
  print('The THETA value: {}'.format(theta))
  print('\nHypothesis Function:')
  print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
  J_histories.append(J_history)
  plot_model_fit(alpha,theta)

alpha_values = np.arange(0.001, 0.02, 0.002)
J_histories =[]

print("Alpha values: ({})".format(len(alpha_values)), alpha_values)
func(0)

func(1)

func(2)

func(3)

func(4)

func(5)

func(6)

func(7)

func(8)

func(9)

x = np.arange(1,125,1.25)
y = alpha_values
len(x)

print(min(min(J_histories)))
max(max(J_histories))

z = J_histories
fig, ax = plt.subplots(1, 1,figsize=(10, 8))
ax.contour(x,y,z)
plt.show()

import plotly.graph_objects as go



fig = go.Figure(data=go.Contour(
	x=x, y=y, z=z,
	contours=dict(
		coloring='lines',
		showlabels=True,),
    line_width=2
))

fig.show()

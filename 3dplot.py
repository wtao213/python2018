# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

#the one doesn't work
data = np.loadtxt(r"C:\Users\wtao\Desktop\AC_CON2.csv", delimiter=',', skiprows=2)

# the one works
data = pd.read_csv(r"C:\Users\wtao\Desktop\neilson_different_channels.csv")

#take a subset of data only contain obs have channel = all channel
data[data.CHANNEL=='ALL CHANNEL']

print(data)

d=pd.Series.tolist(data[data.CHANNEL=='ALL CHANNEL'].PLANNED)


x = pd.Series.tolist(data.PLANNED)
y = pd.Series.tolist(data.AUTOPILOT)
z = pd.Series.tolist(data.Price_Consider)

print(d)

# actual plotting which doesn't work now
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Axes3D.scatter(x, y, z, zdir='z', s=20, c=None,depthshade=True)



#sample for geting a plot
#testing just changes x,y,z value
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#transfer the variable to list
x = pd.Series.tolist(data.PLANNED)
y = pd.Series.tolist(data.AUTOPILOT)
z = pd.Series.tolist(data.Price_Consider)

# add filter for dataset
d=pd.Series.tolist(data[data.CHANNEL=='GROCERY STORE(Conventional)'].PLANNED)
e=pd.Series.tolist(data[data.CHANNEL=='GROCERY STORE(Conventional)'].AUTOPILOT)
f=pd.Series.tolist(data[data.CHANNEL=='GROCERY STORE(Conventional)'].Price_Consider)

#main body for 3d plotting
ax.scatter(d, e, f, c='r', marker='o')

ax.set_xlabel('PLANNED')
ax.set_ylabel('AUTOPILOT')
ax.set_zlabel('Price_Consider')

plt.show()



### second steps, trying to ploting by groups
### import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
 
# Create data
N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N),0.4+0.1*np.random.rand(N))
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N),0.1*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N),0.3*np.random.rand(N))
 
data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water") 

type(data)
print(data)

 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
ax = fig.gca(projection='3d')
 
for data, color, group in zip(data, colors, groups):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
 
plt.title('Matplot 3d scatter plot')
plt.legend(loc=2)
plt.show()

###try to add color for groups







#### actual working here
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

# load the three metrics number
data = pd.read_csv(r"C:\Users\wtao\Desktop\categoryrole.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#check the type of wtp
type(data.WTP)

#transfer the variable to list
x = pd.Series.tolist(data.WTP)
y = pd.Series.tolist(data.CI)
z = pd.Series.tolist(data.CM)

x= data.WTP

#main body for 3d plotting
ax.scatter(x, y, z, c='r', marker='o')


ax.set_xlabel('WTP')
ax.set_ylabel('CI')
ax.set_zlabel('CM')
# set axis range
ax.set_xlim3d(-2,5)
ax.set_ylim3d(0,2)
ax.set_zlim3d(2,27)

plt.show()
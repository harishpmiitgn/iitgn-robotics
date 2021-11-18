import csv
import numpy as np
import math

# This file contains the code for noting the collision points in the initial trajectory


file=open(r"Step_Swing Trajectory_5ft7in.csv")
csvreader = csv.reader(file)

T=[]
Q1=[]
Q2=[]
rows=[]
for row in csvreader:
    rows.append(row)

for i in range(1,282):
    T.append(float(rows[i][0]))
    Q1.append(float(rows[i][1]))
    Q2.append(float(rows[i][2]))

#link lengths:
#Given in the doc
a=43.2 #cm
b=49.2 #cm
# c=8.6 #cm

d=19.5 #cm
e=26.2 #cm

Total_H=a+b
#for measuring height

# for measuring the lowest point in the trajectory
z1=[]                                                              # Denotes points of y coordinates of toe point
z2=[]                                                              # Denotes points of y coordinates of heel point
z=[]                                                               # Array containing the lowest point in trajectory
for i in range(0,281):
    end_effec1=(a*np.cos(math.radians(Q1[i])))+(b*np.cos(math.radians(Q1[i]+Q2[i])))+(d*np.sin(math.radians(Q1[i]+Q2[i])))
    end_effec2=(a*np.cos(math.radians(Q1[i])))+(b*np.cos(math.radians(Q1[i]+Q2[i])))+((e-d)*np.cos(math.radians(90+Q1[i]+Q2[i])))
    z1.append(Total_H-end_effec1)
    z2.append(Total_H-end_effec2)
    if(z1[i]<0 or z2[i]<0):
        z.append(i)
print(z)



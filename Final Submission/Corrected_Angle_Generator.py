import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import scipy

## This files contains the code for calculating the joint angles for generating the trajectory when step height is increased by 2cm

file=open("Step_Swing Trajectory_5ft7in.csv")
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
c=8.6 #cm

d=19.5 #cm
e=26.2 #cm

Total_H=a+b+c
#for measuring height

# for measuring the lowest point in the trajectory
z1=[]                                                                   # Denotes points of y coordinates of toe point
z2=[]                                                                   # Denotes points of y coordinates of heel point
z=[]                                                                    # Array containing the lowest point in trajectory
Lowest_Point=[]
end_effec=[]
x_cordi=[]                                                              # Taking the X-Coordinates of points in the trajectory
for i in range(0,281):
    end_effec1=(a*np.cos(math.radians(Q1[i])))+(b*np.cos(math.radians(Q1[i]+Q2[i])))+(d*np.sin(math.radians(Q1[i]+Q2[i])))
    end_effec2=(a*np.cos(math.radians(Q1[i])))+(b*np.cos(math.radians(Q1[i]+Q2[i])))+((e-d)*np.cos(math.radians(90+Q1[i]+Q2[i])))
    x_cordi.append((a*np.sin(math.radians(Q1[i])))+(b*np.sin(math.radians(Q1[i]+Q2[i])))-(d*np.cos(math.radians(Q1[i]+Q2[i]))))
    end_effec.append(end_effec1)
    z1.append(Total_H-end_effec1)
    z2.append(Total_H-end_effec2)
    if(z1[i]<0 or z2[i]<0):
        z.append(i)


New_Low_Point=[x-2 for x in end_effec]                                  # Corrected Step Height y coordinates

# Making lists for calculating joint Angles
q1_new=[]
q2_new=[]
q2_Final=[]

# To calculate the joint Angles Geometry was Applied, Refer to Figure 1(a), Figure 1(b) for briefly understanding the angles.
alpha=np.arctan2(d,b)
for i in range(281):
    cos = (x_cordi[i]*x_cordi[i] + New_Low_Point[i]*New_Low_Point[i] - a*2 - (b*2+d*2))/(2*a*math.sqrt(b*2+d*2))
    cos = 1 if cos>1 else cos
    cos = -1 if cos<-1 else cos
    q2_new.append(np.arccos(cos))
    q1_new.append((np.arctan2(New_Low_Point[i],x_cordi[i]) - np.arctan2(math.sqrt(b*2+d*2)*np.sin(q2_new[i]),(a + math.sqrt(b*2+d*2)*np.cos(q2_new[i])))))
    q2_Final.append(q2_new[i]+alpha)
print(q1_new)                                                           # Final Q1 (Hip Joint Angle)
print('____________________________________________________--')
print(q2_Final)                                                         # Final Q2 (Knee Joint Angle)


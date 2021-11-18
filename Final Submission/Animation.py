import pandas as pd 
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# This File contains the animation of initial Gait when the hip is stationary, 

# Taking the raw data (as it is) for creating animation and observing the scrubbing points to improve the trajectory further
data = pd.read_csv('Step_Swing Trajectory_5ft7in.csv')
q1 = np.pi/2 - data['LH']*np.pi/180
q2 = data['LK']*np.pi/180
t = data['Time']

# Collision Points, calculated using CollisionPoints.py
coll_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178]

#link lengths:
#Given in the doc
grount_height =92.4                                                 #Height from hip to ground (cm)
l1 = 43.2                                                           #Thigh Length : (cm)
l2 = 40.6                                                           #Knee Length : (cm)
ankle_height = 8.6                                                  #Ankle Length : (cm)
foot_forward_len = 19.5                                             #Front Foot Length
foot_backward_len = 26.2 - 19.5                                     #Back Foot Length

#Plotting the figure for geberating frames
fig,ax = plt.subplots()

# Apending the coordinates for different links
l1_end_positions = []
l2_end_positions = []
l3_end_positions = []
foot_front_ends = []
foot_back_ends = []


for t1,t2 in zip(q1,q2):
    l1_end_position = l1*np.array((np.cos(t1 +np.pi), np.sin(t1+np.pi)))
    l2_end_position = l1_end_position + l2*np.array((np.cos(-np.pi + (t1 - t2)), np.sin(-np.pi + (t1 - t2))))
    l3_end_position = l1_end_position + (l2 + ankle_height)*np.array((np.cos(-np.pi + (t1 - t2)), np.sin(-np.pi + (t1 - t2))))
    foot_back_end = l3_end_position + foot_backward_len*np.array((np.cos(-3*np.pi/2 + (t1 - t2)),np.sin(-3*np.pi/2 + (t1 - t2))))
    foot_front_end = l3_end_position - foot_forward_len*np.array((np.cos(-3*np.pi/2 + (t1 - t2)),np.sin(-3*np.pi/2 + (t1 - t2))))

    l1_end_positions.append(l1_end_position)
    l2_end_positions.append(l2_end_position)
    l3_end_positions.append(l3_end_position)
    foot_front_ends.append(foot_front_end)
    foot_back_ends.append(foot_back_end)

# Various attributes for the animation
ankle_gait_x = [pos[0] for pos in l2_end_positions]
ankle_gait_y = [pos[1] for pos in l2_end_positions]
ax.scatter(ankle_gait_x, ankle_gait_y, s = 0.6, alpha = 0.5)

front_end_gait_x = [pos[0] for pos in l3_end_positions]
front_end_gait_y = [pos[1] for pos in l3_end_positions]
ax.scatter(front_end_gait_x, front_end_gait_y, s = 0.6, alpha = 0.5, color = 'g')

l1_end_circle = plt.Circle([0,0], radius=4, fc='#0077b6', alpha = 0.5)
l2_end_circle = plt.Circle([0,0], radius=4, fc='g',  alpha = 0.5)
l1_line = plt.Line2D([], [], color = "#0077b6" , lw=2.5)
l2_line = plt.Line2D([], [], color = '#0077b6', lw=2.5)
foot_line = plt.Line2D([], [], color = '#0077b6', lw=2.5)
time_text = ax.text(0.02, 0.95, 'time (seconds): 0.00', transform = ax.transAxes)


ax.axhline(-(l1 + l2), alpha = 0)
ax.axhline((l1 + l2), alpha = 0)
ax.axvline((l1 + l2), alpha = 0)
ax.axvline(-(l1 + l2), alpha = 0)
ax.axhline(-grount_height, color = 'g') # Ground Line

ax.add_line(l2_line)
ax.add_patch(foot_line)
ax.add_line(l1_line)
ax.add_patch(l1_end_circle)
ax.add_patch(l2_end_circle)

plt.axis('square')

def animate(i):
    t_ = t[i]
    
    if i in coll_pts:
        foot_line.set_color('r')                                    #plotting the collision points in red
    else:
        foot_line.set_color('#0077b6')
    l1_line.set_data((0,l1_end_positions[i][0]),(0, l1_end_positions[i][1]))
    l2_line.set_data((l1_end_positions[i][0],l3_end_positions[i][0]),(l1_end_positions[i][1],l3_end_positions[i][1]))
    foot_line.set_data((foot_back_ends[i][0], foot_front_ends[i][0]),(foot_back_ends[i][1], foot_front_ends[i][1]))

    l1_end_circle.set_center(l1_end_positions[i])
    l2_end_circle.set_center(l2_end_positions[i])
      
    time_text.set_text('time (seconds): %.2f' % t_)

    return l1_line, l2_line, l1_end_circle, l2_end_circle, foot_line, time_text

anim = animation.FuncAnimation(fig, animate,
                            frames = 280,
                            interval = 2.8,
                            blit = True)
ax.set_title('Stationary Hip Joint: Before Increment')
ax.grid()
plt.show()

# Saving the GIF
writergif = animation.PillowWriter(fps=100) 
anim.save(r'C:\\Users\\videh\\OneDrive\Documents\Sem 5\\ME 639\\Coffee Bots\Workspace\\gifs\\stationary_before.gif', writer = writergif)
import pandas as pd 
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

## This file contains the code when movement of torso is considered and no corrected step height

data = pd.read_csv("Step_Swing Trajectory_5ft7in.csv")
q1 = np.pi/2 - data['LH']*np.pi/180
q2 = data['LK']*np.pi/180
t = data['Time']

#link lengths:
#Given in the doc
ground_height =92.4 #Height from hip to ground (cm)

l1 = 43.2 #Thigh Length : (cm)
l2 = 40.6 #Knee Length : (cm)

ankle_height = 8.6 # (cm)
foot_forward_len = 19.5
foot_backward_len = 26.2 - 19.5


fig,ax = plt.subplots()

hip_positions = []
l1_end_positions = []
l2_end_positions = []
l3_end_positions = []
foot_front_ends = []
foot_back_ends = []


for t1,t2,t3 in zip(q1,q2,t):
    hip_position = 2.5*np.sin(((np.pi/2.8)*t3))

    l1_end_position_x = l1*np.cos(t1 +np.pi)
    l1_end_position_y = l1*np.sin(t1+np.pi) +hip_position
    l1_end_position = np.array([l1_end_position_x,l1_end_position_y])
    l2_end_position_x = l1_end_position_x + l2*np.cos(-np.pi + (t1 - t2))
    l2_end_position_y = l1_end_position_y + l2*np.sin(-np.pi + (t1 - t2))
    l2_end_position = np.array([l2_end_position_x,l2_end_position_y])

    l3_end_position_x = l1_end_position_x + (l2 + ankle_height)*(np.cos(-np.pi + (t1 - t2)))
    l3_end_position_y = l1_end_position_y + (l2 + ankle_height)*(np.sin(-np.pi + (t1 - t2)))
    l3_end_position = np.array([l3_end_position_x,l3_end_position_y])
    
    foot_front_end_x = l3_end_position_x + foot_backward_len*(np.cos(-3*np.pi/2 + (t1 - t2)))
    foot_front_end_y = l3_end_position_y + foot_backward_len*(np.sin(-3*np.pi/2 + (t1 - t2)))
    foot_front_end = np.array([foot_front_end_x,foot_front_end_y])

    foot_back_end_x = l3_end_position_x - foot_forward_len*(np.cos(-3*np.pi/2 + (t1 - t2)))
    foot_back_end_y = l3_end_position_y - foot_forward_len*(np.sin(-3*np.pi/2 + (t1 - t2)))
    foot_back_end = np.array([foot_back_end_x,foot_back_end_y])

    hip_positions.append(hip_position)
    l1_end_positions.append(l1_end_position)
    l2_end_positions.append(l2_end_position)
    l3_end_positions.append(l3_end_position)
    foot_front_ends.append(foot_front_end)
    foot_back_ends.append(foot_back_end)


gait_x = [pos[0] for pos in foot_back_ends]
gait_y = [pos[1] for pos in foot_back_ends]
ax.scatter(gait_x, gait_y, s = 0.6, alpha = 0.5)

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
ax.axhline(-ground_height, color = 'g') # Ground Line

ax.add_line(l2_line)
ax.add_patch(foot_line)
ax.add_line(l1_line)
ax.add_patch(l1_end_circle)
ax.add_patch(l2_end_circle)

plt.axis('square')

coll_points = []
def animate(i):
    t_ = t[i]
     
    l1_line.set_data((0,l1_end_positions[i][0]),(hip_positions[i], l1_end_positions[i][1]))
    l2_line.set_data((l1_end_positions[i][0],l3_end_positions[i][0]),(l1_end_positions[i][1],l3_end_positions[i][1]))
    foot_line.set_data((foot_back_ends[i][0], foot_front_ends[i][0]),(foot_back_ends[i][1], foot_front_ends[i][1]))

    l1_end_circle.set_center(l1_end_positions[i])
    l2_end_circle.set_center(l2_end_positions[i])
      
    time_text.set_text('time (seconds): %.2f' % t_)
    
    if (foot_front_ends[i][1] < -ground_height) or (foot_back_ends[i][1] < -ground_height):
        foot_line.set_color('r')
        coll_points.append(i)
    else:
        foot_line.set_color('#0077b6')

    return l1_line, l2_line, l1_end_circle, l2_end_circle, foot_line, time_text

anim = animation.FuncAnimation(fig, animate,
                            frames = 280,
                            interval = 2.8,
                            blit = True)

ax.set_title('Moving Hip Joint: Before Increment')
ax.scatter([x[0] for x in foot_front_ends], [x[1] for x in foot_front_ends], s = 0.6, color = 'g')

preet_hope =[]

for i, ((x_f,y_f), (x_b, y_b))  in enumerate(zip(foot_back_ends, foot_front_ends)):
    if y_b <= y_f:
        print(i)
        preet_hope.append((x_b,y_b))
    else:
        preet_hope.append((x_f,y_f))

print(preet_hope)

ax.grid()
plt.show()
# writergif = animation.PillowWriter(fps=100) 
# anim.save(r'C:\\Users\\videh\\OneDrive\Documents\Sem 5\\ME 639\\Coffee Bots\Workspace\\gifs\\hip_move_before.gif', writer = writergif)
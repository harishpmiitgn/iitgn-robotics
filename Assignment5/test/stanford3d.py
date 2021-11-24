import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import mpl_toolkits.mplot3d.art3d as a3

q1 = 0
q2 = 0
d3 = 1
d3_max = 6

l1 = 5
l2 = 5
l3 = 2

#Initialising Animation
plt.ion()

fig = plt.figure()
ax = p3.Axes3D(fig, auto_add_to_figure = False)
fig.add_axes(ax)

link1_line = ax.plot([0,0], [0,0], [0,l1])[0]
link2_line = ax.plot(np.array([0, l2*np.cos(q2)*np.cos(q1)]),np.array([0, l2*np.cos(q2)*np.sin(q1)]),np.array([l1, l1+l2*np.sin(q2)]))[0]
link3_line = ax.plot(np.array([l2*np.cos(q2)*np.cos(q1),(l2+l3)*np.cos(q2)*np.cos(q1)]), np.array([l2*np.cos(q2)*np.sin(q1),(l2+l3)*np.cos(q2)*np.sin(q1)]), np.array([l1+l2*np.sin(q2),l1+(l2+l3)*np.sin(q2)]))[0]
extension_line = ax.plot(np.array([(l2+l3)*np.cos(q2)*np.cos(q1),(l2+l3+d3)*np.cos(q2)*np.cos(q1)]), np.array([(l2+l3)*np.cos(q2)*np.sin(q1),(l2+l3+d3)*np.cos(q2)*np.sin(q1)]), np.array([l1+(l2+l3)*np.sin(q2),l1+(l2+l3+d3)*np.sin(q2)]))[0]

# boundary lines
ax.set_xlim3d([-(l2+l3+d3_max), (l2+l3+d3_max)])
ax.set_xlabel('X')

ax.set_ylim3d([-(l2+l3+d3_max), (l2+l3+d3_max)])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 2*(l2+l3+d3_max)])
ax.set_zlabel('Z')

ax.set_title('Stanford Manipulator')


def anim_update(q2_ = np.pi/4):
    # q1 = np.pi/4
    q2 = q2_

    link2_line.set_data_3d(np.array([0, l2*np.cos(q2)*np.cos(q1)]),np.array([0, l2*np.cos(q2)*np.sin(q1)]),np.array([l1, l1+l2*np.sin(q2)]))
    link3_line.set_data_3d(np.array([l2*np.cos(q2)*np.cos(q1),(l2+l3)*np.cos(q2)*np.cos(q1)]), np.array([l2*np.cos(q2)*np.sin(q1),(l2+l3)*np.cos(q2)*np.sin(q1)]),np.array([l1+l2*np.sin(q2),l1+(l2+l3)*np.sin(q2)]))
    extension_line.set_data_3d(np.array([(l2+l3)*np.cos(q2)*np.cos(q1),(l2+l3+d3)*np.cos(q2)*np.cos(q1)]), np.array([(l2+l3)*np.cos(q2)*np.sin(q1),(l2+l3+d3)*np.cos(q2)*np.sin(q1)]),np.array([l1+(l2+l3)*np.sin(q2),l1+(l2+l3+d3)*np.sin(q2)]))
    
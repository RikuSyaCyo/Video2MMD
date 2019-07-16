import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import imageio

def loadDepthData(fileName):
	depth_info = []
	with open(fileName) as textData:
		lines = textData.readlines()
		for line in lines:
			lineData = line.strip('\r\n').split(',')
			depth_info.append([int(lineData[0]), float(lineData[1].strip(' '))])
	return depth_info

def loadPoseData(fileName):
    pose_info = []
    index = 0
    with open(fileName) as textData:
        lines = textData.readlines()
        for line in lines:
            lineData = line.strip('\r\n').split(',')
            pose_info.append([])
            for data in lineData:
                if len(data) != 1:
                    joint_data = data.strip(' ').split(' ')
                    joint_data = list(map(lambda x: float(x), joint_data))
                    pose_info[index].append(joint_data)
            index += 1
    return pose_info

def show3Dpose(p3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False, root_xyz=None): # blue, orange


  I   = np.array([0,4,5,0,1,2,0,7,8,9,8,14,15,8,11,12]) # start points
  J   = np.array([4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]) # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  # Make connection matrix
  for i in np.arange( len(I) ):
   #x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    #print(p3d[0][1])
    x, y, z = [np.array([p3d[I[i]][j], p3d[J[i]][j]]) for j in range(3)]
    
    ax.plot(x, y, z, marker='o', markersize=2, lw=1, c=lcolor if LR[i] else rcolor)
    #ax.grid(True)

  if root_xyz is not None:
    ax.set_xlim3d([-1500+root_xyz[0], 1500+root_xyz[0]])
    ax.set_zlim3d([-1000+root_xyz[2], 1000+root_xyz[2]])
    ax.set_ylim3d([-1200+root_xyz[1], 1200+root_xyz[1]])
  else:
    RADIUS = 750 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels
  # ax.set_xticks([])
  # ax.set_yticks([])
  # ax.set_zticks([])

  # ax.get_xaxis().set_ticklabels([])
  # ax.get_yaxis().set_ticklabels([])
  # ax.set_zticklabels([])
  #ax.set_aspect('equal')
  ax.set_aspect('auto')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  #ax.w_xaxis.set_pane_color(white)
  #ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  #ax.w_xaxis.line.set_color(white)
  #ax.w_yaxis.line.set_color(white)
  #ax.w_zaxis.line.set_color(white)

def smoothing_depth(d, dw):
  if len(dw) < frame:
    dw.append(d)
    dw_sorted = sorted(dw)
    if len(dw) % 2 != 1:
      dw_sorted.append(d)
  else:
    dw.append(d)
    dw = dw[1:]
    dw_sorted = sorted(dw)
  #print(dw_sorted, dw_sorted[int(len(dw_sorted)/2)])
  return [dw, dw_sorted[int(len(dw_sorted)/2)]]

depth_data = loadDepthData('./depth.txt')
pose_data = loadPoseData('./pos.txt')
#print(pose_data[0][1])

def depth_visualization(depth, depth_new):
	x = []
	count = 0
	for frame in depth:
		count += 1
		x.append(count)
	plt.figure(figsize=(8,4))
	plt.plot(x, depth, "b--",linewidth=1)
	plt.plot(x, depth_new, "r--", linewidth=1)
	plt.xlabel("frames")
	plt.ylabel("depth")
	plt.title("Depth Smoothed")
	#plt.show()
	plt.savefig("line.jpg")


gs1 = gridspec.GridSpec(1, 1)
gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
plt.axis('on')
subplot_idx, exidx = 1, 1

pngName_list = []
index = 0
first_xyz = [0,0,0]
f = open('./global_pos.txt','a+')

depth_min = min(list(map(lambda x: x[1], depth_data)))
depth_max = max(list(map(lambda x: x[1], depth_data)))
#print(depth_min, depth_max)

depth_new = []
depth_old = []
depth_window = []
frame = 15
for joint_all in pose_data:
    ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
    ax.grid(True)
    ax.view_init(18, 280)

    p3d = []
    if index > len(depth_data) - 1:
        break
    depth_0 = depth_data[index][1]

    depth_scaled = 3000 * (depth_0 - depth_min) / (depth_max - depth_min)
    depth_window, depth_smoothed = smoothing_depth(depth_scaled, depth_window)
    depth_old.append(depth_scaled)
    depth_new.append(depth_smoothed)
    print('original depth: ' + str(depth_0) + ' |smoothed depth: ' + str(depth_smoothed))

    for joint in joint_all:
        joint[2] -= depth_smoothed
        p3d.append([joint[1], joint[2], joint[3]])
    f.write(str(p3d))
    f.write('\n')
    index += 1
    #print(p3d)
    if index == 1:
        first_xyz = [p3d[0][0], 0, p3d[0][2]]
    show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True, root_xyz=first_xyz)
    pngName = './result_png/' + str(index) + '.png'
    plt.savefig(pngName)
    pngName_list.append(imageio.imread(pngName))

depth_visualization(depth_old, depth_new)
print('saving gif to local file')
imageio.mimsave('movie_smoothing.gif', pngName_list, fps=60)




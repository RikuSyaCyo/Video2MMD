from numpy import *

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
    with open(fileName) as textData:
        lines = textData.readlines()
        for line in lines:
            lineData = line.strip('\r\n').split(',')
            for data in lineData:
                if len(data) != 1:
                    joint_data = data.strip(' ').split(' ')
                    joint_data = list(map(lambda x: float(x), joint_data))
                    pose_info.append(joint_data)
    return pose_info

depth_data = loadDepthData('./depth.txt')
pose_data = loadPoseData('./pos.txt')
# print(pose_data)




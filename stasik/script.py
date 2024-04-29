print(1)
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
cloud = o3d.io.read_point_cloud("D:/university/6semestr/AI-diploma-niias/stasik/src/cloud_2_0100.pcd")

print(np.asarray(cloud.points))
print(np.asarray(cloud.colors))
print(np.asarray(cloud.normals))
print(cloud)
print('Shape of points', np.asarray(cloud.points).shape)

points = np.asarray(cloud.points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt

cloud_path = '../vkr/safety-doors/08_00 МСК 23 мая/point _end/clouds_stereo/cloud_2_0001.pcd'
json_path = '../vkr/safety-doors/08_00 МСК 23 мая/point _end/clouds_stereo_ann/cloud_2_0001.pcd.json'

cloud_path = '../vkr/safety-doors/08_00 МСК 23 мая/point _end/clouds_stereo/cloud_2_0010.pcd'
json_path = '../vkr/safety-doors/08_00 МСК 23 мая/point _end/clouds_stereo_ann/cloud_2_0010.pcd.json'

cloud_path = '../../vkr/safety-doors/08_00 МСК 23 мая/point _end/clouds_tof/cloud_0_0041.pcd'
json_path = '../../vkr/safety-doors/08_00 МСК 23 мая/point _end/clouds_tof_ann/cloud_0_0041.pcd.json'

cloud = o3d.io.read_point_cloud(cloud_path)
print('points:', np.asarray(cloud.points))
print('colors:', np.asarray(cloud.colors))
print('normals:', np.asarray(cloud.normals))
print(cloud)

cuboid_colors = {
    'human': [1, 0.2, 0.2],
    'wear': [0.2, 1, 0.2],
    'other': [0.2, 1, 1],
    'limb': [1, 1, 0.2]
}

def create_cuboid(cub, data):
    position = cub['geometry']['position']
    rotation = cub['geometry']['rotation']
    dimensions = cub['geometry']['dimensions']
    # Создание кубоида
    width, height, depth = dimensions['x'], dimensions['y'], dimensions['z']
    cuboid_3d = o3d.geometry.TriangleMesh.create_box(width, height, depth)

    for obj in data['objects']:
        if obj['key'] == cub['objectKey']:
            classTitle = obj['classTitle']
    cuboid_3d.paint_uniform_color(cuboid_colors[classTitle])
    x, y, z = position['x'], position['y'], position['z']
    x -= 0.5 * width
    y -= 0.5 * height
    z -= 0.5 * depth
    cuboid_3d.translate([x, y, z])

    alfa, beta, gamma = rotation['x'], rotation['y'], rotation['z']
    rotate = cuboid_3d.get_rotation_matrix_from_xyz((alfa, beta, gamma))
    cuboid_3d.rotate(rotate)

    return cuboid_3d

with open(json_path) as f:
    data = json.load(f)
figures = data['figures']

cuboids = [create_cuboid(figure, data) for figure in figures]

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[0, 0, 0])

cloud_downsampled = cloud.voxel_down_sample(voxel_size=0.05)
# cloud_downsampled = cloud.uniform_down_sample(every_k_points=5)

w, index = cloud_downsampled.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
print('Fitted: ', w)
non_floor_cloud = cloud_downsampled.select_by_index(index, invert=True)


# o3d.visualization.draw_geometries([cloud, axes] + cuboids)
o3d.visualization.draw_geometries([cloud, axes])
# o3d.visualization.draw_geometries([non_floor_cloud, axes])

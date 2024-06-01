import pandas as pd
import numpy as np
import scipy

import json

import open3d as o3d

from tqdm import tqdm
import os

from typing import List, Dict,Tuple,Any

import catboost
from catboost import CatBoostClassifier, Pool

import sklearn
from sklearn.model_selection import train_test_split,KFold,GroupKFold,StratifiedGroupKFold,StratifiedKFold

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

object_classes = {
    'limb': 0,
    'other': 1,
    'wear': 2,
    'human': 3
}
object_classes_color_map = {
    0: (0, 1, 0),
    1: (0.2, 0.2, 0.2),
    2: (1, 0, 0),
    3: (0, 0, 1)
}

knn = 40
radius = 0.15

# train_dataframe = pd.read_csv(f'test_{knn}_{radius}.csv')

import math

class PointCloudDataset(object):
    def __init__(self, points_path: str, ann_path: str, knn: int, radius: float, min_neighbours):
        """
            In :
              data_path: str - путь до папки с данными
              grouping_method : str - метод поиска соседей , ["knn","radius_search",имплементированный вами]
              neighbourhood_th : Any[int,float] - пороговое значение для k - количества соседей или radius - радиуса сферы
        """

        self.points_path = points_path
        self.ann_path = ann_path
        self.knn = knn
        self.radius = radius
        self.min_neighbours = min_neighbours

        self.feature_names = ['x', 'y', 'z', 'intensity', 'eigenvals_sum', 'linearity', 'planarity', 'change_of_curvature',
                            'scattering', 'omnivariance', 'anisotropy', 'eigenentropy', 'label','scene_id']

    def create_cuboid(cub):
        position = cub['geometry']['position']
        rotation = cub['geometry']['rotation']
        dimensions = cub['geometry']['dimensions']
        # Создание кубоида
        extent = np.array([dimensions['x'], dimensions['y'], dimensions['z']])
        center = np.array([position['x'], position['y'], position['z']])
        euler = np.array([rotation['x'], rotation['y'], rotation['z']])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(euler[0]), -np.sin(euler[0])],
            [0, np.sin(euler[0]), np.cos(euler[0])]
        ])
        Ry = np.array([
            [np.cos(euler[1]), 0, np.sin(euler[1])],
            [0, 1, 0],
            [-np.sin(euler[1]), 0, np.cos(euler[1])]
        ])
        Rz = np.array([
            [np.cos(euler[2]), -np.sin(euler[2]), 0],
            [np.sin(euler[2]), np.cos(euler[2]), 0],
            [0, 0, 1]
        ])
        rot = np.linalg.inv(Rx @ Ry @ Rz)

        return center, extent / 2, rot, cub['objectKey']

    def read_scene_from_file(self, points_file, ann_file) -> Tuple[np.ndarray,np.ndarray]:
        cloud = o3d.io.read_point_cloud(points_file)
        # cloud = cloud.voxel_down_sample(voxel_size=0.025)
        # cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)[:, 0]

        with open(ann_file) as f:
            json_data = json.load(f)
            cuboids = [PointCloudDataset.create_cuboid(x) for x in json_data['figures']]
            objects = dict((x['key'], x['classTitle']) for x in json_data['objects'])

        labels = []
        for point in points:
            label = object_classes['other']
            for center, hextent, rot, key in cuboids:
                point_local = np.dot(point[:3] - center, rot)
                if np.all(np.abs(point_local) <= hextent):
                    label = object_classes[objects[key]]
                    break
            labels.append(label)
        return points, colors, labels

    def load_from_directory(self, start, end) -> Tuple[List[np.ndarray],List[np.ndarray]]:
        for points_file, ann_file in zip(sorted(os.listdir(self.points_path))[start:end], sorted(os.listdir(self.ann_path))[start:end]):
            points_file = os.path.join(self.points_path, points_file)
            ann_file = os.path.join(self.ann_path, ann_file)

            yield self.read_scene_from_file(points_file, ann_file)

    def create_kdtree(self, points: np.ndarray)-> Tuple[o3d.geometry.PointCloud,o3d.geometry.KDTreeFlann]:
        xyz = o3d.utility.Vector3dVector(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = xyz

        tree = o3d.geometry.KDTreeFlann(pcd)

        return pcd, tree

    def rknn_search(self, pcd: o3d.geometry.PointCloud, tree: o3d.geometry.KDTreeFlann, query_index: int, k: int, radius: float) -> np.ndarray:
        _, indices, _ = tree.search_hybrid_vector_3d(pcd.points[query_index], radius, k)

        return np.array(pcd.points)[indices]

    def get_eugen_stats(self, neighbourhood_points: np.ndarray) -> Tuple[float, ...]:
        centered = neighbourhood_points - np.mean(neighbourhood_points, axis=0)[None, :]
        cov = np.cov(centered.T)
        assert cov.shape == (3, 3)
        e, v = np.linalg.eig(cov)
        e = np.real(e)[::-1]
        s = np.sum(e)

        e = np.clip(e, 0, None) + 1e-6

        sum_of_eigenvalues = s
        linearity = (e[0] - e[1]) / e[2]
        planarity = (e[1] - e[2]) / e[0]
        scattering = e[2] / e[0]
        omnivariance = 3 * math.sqrt(e[0] * e[1] * e[2])
        anisotropy = (e[0] - e[2]) / e[0]
        eigentropy = -(e[0] / s + math.log(e[0] / s)) - (e[1] / s + math.log(e[1] / s)) - (e[2] / s + math.log(e[2] / s))
        change_of_curvature = (e[0] - e[1]) / s

        return sum_of_eigenvalues, linearity, planarity, change_of_curvature, scattering, omnivariance, anisotropy, eigentropy

    def create_dataset(self, start, end) -> pd.DataFrame:
        data = []
        for scene_id, (points, colors, labels) in tqdm(enumerate(self.load_from_directory(start, end)), desc='Scene processing'):
            pcd, tree = self.create_kdtree(points)

            for i, (point, color, label) in enumerate(zip(points, colors, labels)):
                neighbors = self.rknn_search(pcd, tree, i, self.knn, self.radius)

                if len(neighbors) < self.min_neighbours or point[2] < 3:
                  continue

                features = self.get_eugen_stats(neighbors)

                point_data = list(point) + [color] + list(features) + [label, scene_id+1]
                data.append(point_data)

        dataframe = pd.DataFrame(data, columns=self.feature_names)
        return dataframe

def visualize_point_cloud(points: np.ndarray, labels: np.ndarray, smoothen=False) -> go.Figure:
    if not smoothen:
        colors = [object_classes_color_map[label] for label in labels]
    else:
        xyz = o3d.utility.Vector3dVector(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = xyz
        tree = o3d.geometry.KDTreeFlann(pcd)

        k = 10
        colors = []
        for i, point in enumerate(points):
            _, indices, _ = tree.search_knn_vector_3d(point, k)
            values, counts = np.unique(labels[np.array(indices)], return_counts=True, axis=0)
            # print(values, counts, len(indices))
            # assert np.all(values[np.argmax(counts)] != np.array([0, 0, 0]))
            label = values[np.argmax(counts)]
            colors.append(object_classes_color_map[label])

    fig = go.Figure(data=[go.Scatter3d(x=points[:,0],y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(color=colors, size=5))])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        zaxis=dict(range=[-1, 0], autorange='reversed')
    ))
    return fig

train_data_path = "../vkr/safety-doors/08_00 МСК 23 мая/point _end" # Путь до тренировочных данных
# train_data_path = '/content/drive/MyDrive/safety-doors/08_00 МСК 23 мая-20240305T095937Z-001/08_00 МСК 23 мая/point _end/point _end'
points_path = os.path.join(train_data_path, 'clouds_tof')
ann_path = os.path.join(train_data_path, 'clouds_tof_ann')

train_dataset = PointCloudDataset(points_path, ann_path, knn, radius, knn//4)
train_dataframe = train_dataset.create_dataset(46, 47)

# scene_0 = train_dataframe[train_dataframe["scene_id"] == 47]
scene_0 = train_dataframe
points  = scene_0.to_numpy()[:,:3]
labels = scene_0.to_numpy()[:,-2]

visualize_point_cloud(points, labels).show()

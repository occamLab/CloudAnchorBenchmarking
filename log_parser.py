
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import os

class LogParser:
    def __init__(self):
        self.all_logs = sorted(glob.glob('logs/*.log'), key=os.path.getmtime, reverse=True)
        self.anchor_poses = {}

    def plot_path(self, pose_matrices):
        fig = plt.figure()
        ax = plt.axes()
        plt.scatter(pose_matrices[:,2,3], pose_matrices[:,0,3])
        ax.axis('equal')
        return fig, ax

    def analyze_file(self, file_path, create_plots=True):
        print(file_path)
        self.anchor_poses = {}
        with open(file_path) as f:
            d = json.load(f)
        resolutions = d['cloudAnchorResolutions'] if 'cloudAnchorResolutions' in d else []
        cloud_landmarks = d['cloudAnchorLandmarks'] if 'cloudAnchorLandmarks' in d else []
        poses = d['poses']
        resolved_anchor_set = set(map(lambda x: x['cloudID'], resolutions))
        unresolved_anchors = set(cloud_landmarks) - resolved_anchor_set
        pose_matrices = np.asarray(poses).reshape(-1, 4, 4).swapaxes(1,2)
        if create_plots:
            fig, ax = self.plot_path(pose_matrices)
            plt.show()

        for cloudID in resolved_anchor_set:
            fig = self.plot_path(pose_matrices)
            anchor_resolutions = list(filter(lambda x: x['cloudID'] == cloudID, resolutions))
            self.anchor_poses[cloudID] = np.asarray(list(map(lambda x: x['pose'], anchor_resolutions))).reshape(-1, 4, 4).swapaxes(1, 2)
            map_poses = np.asarray(list(map(lambda x: x['mapPose'], anchor_resolutions))).reshape(-1, 4, 4).swapaxes(1, 2)
            print(self.anchor_poses[cloudID].shape)
            if create_plots:
                plt.scatter(self.anchor_poses[cloudID][:,2,3], self.anchor_poses[cloudID][:,0,3], color='k')
                plt.quiver(self.anchor_poses[cloudID][:,2,3],
                           self.anchor_poses[cloudID][:,0,3],
                           self.anchor_poses[cloudID][:,0,0],
                           self.anchor_poses[cloudID][:,2,0], color='r')
                plt.quiver(self.anchor_poses[cloudID][:,2,3],
                           self.anchor_poses[cloudID][:,0,3],
                           self.anchor_poses[cloudID][:,0,2],
                           self.anchor_poses[cloudID][:,2,2],
                           color='y')
                plt.legend(['path', 'cloud anchor positions', 'cloud anchor x-axis', 'cloud anchor z-axis'])
                plt.show()
        return (file_path, resolved_anchor_set, unresolved_anchors)
    
    def get_anchor_poses(self):
        return self.anchor_poses

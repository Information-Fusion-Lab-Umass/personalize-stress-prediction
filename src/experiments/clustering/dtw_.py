# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# file_name: dtw_.py
# Functionality: Class, DTW_clusters: do clustering based on DTW distance
# Author: Yunfei Luo
# Start date: EST Mar.25th.2020
# Last update: EST Apr.8th.2020
# ----------------------------------------------------------------

import numpy as np
import src.experiments.clustering.density_based_clustering as dbc

class DTW_clusters:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = 0
        self.dist_matrix = list() # 2D array of distance matrix
        self.groups = dict() # dictionary, map: pts_ind -> group
        self.pts = list()
    
    # helper functions
    def cluster_by_construct_graph(self):
        '''
        Construct graph where each node represent each data point;
        Add Edge between two nodes if their distance is below eps T;
        Collect cluster information by retrieve connected graphs.
        '''
        # helper function
        def dfs(ind, choosen, group_id):
            for i in range(len(choosen)):
                if choosen[i]:
                    continue
                if self.dist_matrix[ind][i] <= self.eps:
                    choosen[i] = True
                    self.groups[i] = group_id
                    dfs(i, choosen, group_id)

        # group the data points w.r.t eps
        choosen = [False for _ in range(len(self.pts))]
        group_id = -1
        while False in choosen:
            group_id += 1
            ind = choosen.index(False)
            choosen[ind] = True
            self.groups[ind] = group_id
            dfs(ind, choosen, group_id)
    
    def density_based_clustering(self):
        '''
        clusteri by DBSCAN or OPTICS (xi)
        '''
        group_assign = dict()
        #clusters = OPTICS(min_samples=2, max_eps=75, cluster_method='xi', metric='precomputed').fit(self.dist_matrix)
        clusters = dbc.density_based_clustering(eps=self.eps, min_samples=self.min_samples, cluster_method='xi', metric='precomputed').fit(self.dist_matrix)
        
        for i in range(len(self.pts)):
            self.groups[i] = clusters.labels_[i]
            try:
                group_assign[clusters.labels_[i]].append(self.pts[i])
            except:
                group_assign[clusters.labels_[i]] = [self.pts[i]]
        
        # # visualize
        # import matplotlib.pyplot as plt
        # for pt in group_assign[6]:
        #     plt.plot([i[0] for i in pt], [i[1] for i in pt])
        # plt.show()

    def fit(self, data):
        # calculate distance matrix
        dist_matrix = list()
        for pt1 in data:
            row = list()
            for pt2 in data:
                row.append(self.dtw_dist(pt1, pt2))
            dist_matrix.append(row)
        self.dist_matrix = np.array(dist_matrix)
        self.pts = data

        # plt.imshow(self.dist_matrix, cmap='gray')
        # plt.show()
        
        self.density_based_clustering()

        return self
    
    def predict(self, pts):
        res = list()
        for pt in pts:
            group = -1
            min_dist = np.inf
            for i in range(len(self.pts)):
                curr_dist = self.dtw_dist(pt, self.pts[i])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    group = self.groups[i]
            res.append(group)
        return res

    # helper functions
    # Calculate distance between to points
    def dist(self, p1, p2):
        return np.linalg.norm(p1-p2, ord=2)

    # Calculate DTW distance between to series data
    def dtw_dist(self, ts1, ts2):
        DTW = dict()
        DTW[(0, 0)] = 0
        
        for i in range(len(ts1)):
            for j in range(len(ts2)):
                if i == 0 and j == 0:
                    continue
                cost = self.dist(ts1[i], ts2[j])
                min_ = None
                if i - 1 >= 0 and j - 1 >= 0:
                    min_ = min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
                elif i - 1 >= 0:
                    min_ = DTW[(i-1, j)]
                elif j - 1 >= 0:
                    min_ = DTW[(i, j-1)]
                DTW[(i, j)] = cost + min_
        
        return DTW[(len(ts1) - 1, len(ts2) - 1)]

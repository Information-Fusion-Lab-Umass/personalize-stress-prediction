# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# file_name: density_based_clustering.py
# Functionality: Class, density_based_clustering: do clustering based on density
# Author: Yunfei Luo
# Start date: EST Apr.9th.2020
# Last update: EST Apr.9th.2020
# ----------------------------------------------------------------

import numpy as np
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN

class density_based_clustering:
    def __init__(self, eps, min_samples, cluster_method, metric):
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_method = cluster_method # xi or dbscan
        self.metric = metric
        self.dist_matrix = list()
        self.labels_ = list()

    def fit(self, pts):
        # clustering
        self.pts = np.array(pts)
        if self.metric == 'precomputed':
            self.dist_matrix = self.pts
            print('############################ Predicted by DBSCAN ############################')
            #clusters = OPTICS(min_samples=self.min_samples, max_eps=self.eps, cluster_method=self.cluster_method, metric='precomputed').fit(self.dist_matrix)
            clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed').fit(self.dist_matrix)
            self.labels_ = clusters.labels_
        else:
            self.dist_matrix = self.compute_dist_matrix(self.pts)
            print('############################ Predicted by DBSCAN ############################')
            #clusters = OPTICS(min_samples=self.min_samples, max_eps=self.eps, cluster_method=self.cluster_method, metric='precomputed').fit(self.dist_matrix)
            clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed').fit(self.dist_matrix)
            self.labels_ = clusters.labels_
        
        # # breif view
        # for row in self.dist_matrix:
        #     print(row)

        # clustering outliers
        outliers_ind = [i for i in range(len(clusters.labels_)) if clusters.labels_[i] == -1]
        print('Num of outliers: ', len(outliers_ind))
        self.cluster_outliers(outliers_ind)
        
        return self

    def cluster_outliers(self, outliers_ind):
        '''
        a. Put each outlier to the cluster where its closest points belongs to;
        b. Repeat a. until no change made;
        c. Treat the rest outliers as in one group;
        '''
        # Cluster the outliers that are close to the exist clusters
        hasChange = True
        while hasChange:
            hasChange = False
            for i in outliers_ind:
                if self.labels_[i] != -1:
                    continue
                min_ind = -1
                min_dist = np.inf
                for j in range(len(self.pts)):
                    if i != j:
                        if self.dist_matrix[i][j] < min_dist:
                            min_dist = self.dist_matrix[i][j]
                            min_ind = j
                if self.labels_[min_ind] != -1:
                    self.labels_[i] = self.labels_[min_ind]
                    hasChange = True
        
        # cluster the rest outliers
        outliers_ind = [i for i in outliers_ind if self.labels_[i] == -1]
        left = [i for i in outliers_ind]
        while len(left) > 0:
            group_id = max(self.labels_) + 1
            i = left[0]
            self.labels_[i] = group_id

            while True:
                # find closest node in outliers
                min_ind = -1
                min_dist = np.inf
                for j in outliers_ind:
                    if i != j:
                        if self.dist_matrix[i][j] < min_dist:
                            min_dist = self.dist_matrix[i][j]
                            min_ind = j
                
                # if the closest is grouped, break
                if len(left) == 1 or self.labels_[min_ind] != -1:
                    if len(left) == 1:
                        print('hey here')
                    self.labels_[i] = self.labels_[min_ind]
                    break
                else:
                    self.labels_[min_ind] = group_id
                i = min_ind
            
            # update ouliers list (delete those with group)
            left = [i for i in left if self.labels_[i] == -1]
            

        '''
        ## pseudo-code
        
        loop:
            for each outliers i:
                find i's closest point j
                if j is in an existed group:
                    put i into j's group
            if at least 1 outliers being grouped:
                continue
            else:
                break the loop
        
        # # after the above loop, all the outliers' closest points are also outliers
        # # group them greedly:

        while still have outliers:
            choose the first point i, put it in a new group k
            loop: 
                find i's closest point j
                if j haven't grouped:
                    put j into the group k
                else:
                    put i to the group of j
                    break
                i = j
            update outliers list
        '''

    # helper functions
    # Calculate distance between to points
    def compute_dist_matrix(self, pts):
        compute_dist = None
        if self.metric == 'l1':
            compute_dist = lambda x1, x2: np.linalg.norm(x1-x2, 1)
        elif self.metric == 'l2':
            compute_dist = lambda x1, x2: np.linalg.norm(x1-x2, 2)
        elif self.metric == 'dtw':
            compute_dist = self.dtw_dist
        else:
            print('The distance computing type is not available yet...')
            exit()
        dist_matrix = list()
        for pt1 in pts:
            row = list()
            for pt2 in pts:
                row.append(compute_dist(pt1, pt2))
            dist_matrix.append(row)
        return np.array(dist_matrix)  

    ## DTW distance ##############################################################
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
    ##############################################################################
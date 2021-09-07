# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:28:45 2021

@author: abhis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
import scipy.cluster.hierarchy as shc

class agglomerative:
    def __init__(self,data):
         self.classes={}
         self.list=[]
         self.data=data
    def intial_clusters(self):
        #initially all points are clusters
        for i in range(self.row_len):
            self.list.append(i)
        self.classes[0]=self.list.copy()
        
    def find_minvalue_index(self,matrix):
        #finding min value in matrix 
            self.min1=sys.maxsize
            for p in range(self.row_len):
                for q in range(self.col_len):
                    if p!=q and self.min1>matrix[p][q]:
                        self.min1=matrix[p][q]
                        self.rindex=p
                        self.cindex=q
        
    def update_distace_matrix(self,matrix):
        #updating the matrix after finding min distance using complete linkage        
            self.min_index=min(self.rindex,self.cindex)
            self.max_index=max(self.rindex,self.cindex)            
            
            for i in range(self.row_len):                    
                    if i!=self.min_index and i!=self.max_index:
                        val=max(matrix[self.min_index][i],matrix[self.max_index][i])
                        matrix[self.min_index][i]=val
                        matrix[i][self.min_index]=val
            
            #putting infinite in row and column with higher index
            for i in range(self.row_len):
                matrix[self.max_index][i]=sys.maxsize
                matrix[i][self.max_index]=sys.maxsize
                
    def update_index_of_new_datapoints(self):               
        for i in range(len(self.list)):
            if self.list[i]==self.max_index:
                self.list[i]=self.min_index
    
    #function to cut the dendogram or to find out in which  data is present in cluster 
    #when no of cluster=4     
    def generate_clusters(self):
        self.cluster={} 
        data_points_as_index = self.classes[0]            
        array_of_cluster_points = set(data_points_as_index)
        array_of_cluster_points1=list(array_of_cluster_points) 
        
        for p in range(1,self.row_len):                
            data_points_as_index = self.classes[p]            
            array_of_cluster_points = set(data_points_as_index)
            array_of_cluster_points1=list(array_of_cluster_points) 
            if len(array_of_cluster_points)==4:
                break
                 
        print("Cluster centroids datapoint index",array_of_cluster_points)    
        for i in range(len(array_of_cluster_points1)):
                self.cluster[i]=[]
                for j in range(len(data_points_as_index)):
                    if array_of_cluster_points1[i]==data_points_as_index[j]:
                        self.cluster[i].append(self.data[j]) 
               
    #fit method
    def fit(self,matrix):        
        self.row_len=matrix.shape[0]
        self.col_len=matrix.shape[1]
        
        self.intial_clusters()
        
        for itr in range(1,self.row_len):            
            #finding min value in distance matrix and storing its row and column index
            self.find_minvalue_index(matrix)
            
            #combining these two datapoints in single cluster and updating distance matrix
            #using average linkage
            self.update_distace_matrix(matrix)
            
            #updating same index of all the points present in same cluster in list[]
            self.update_index_of_new_datapoints()
            
            #storing the list which contains data points as index and cluster no as its value
            #storing that list in classes dictionary to keep track of clusters in every iteration
            self.classes[itr]=self.list.copy()
           
        #calling function to generate clusters    
        self.generate_clusters()   
    
    
    
def main():
    df=pd.read_csv('BR_mod.csv')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(),inplace=True)
    X=np.array(df.iloc[:,:].astype(float))
    
    intial_dst_matrix=pairwise_distances(X,metric='euclidean')  
   
    #uodating diagonal elements as infinity so that it is easy to find min value in distance matrix
    for i in range(intial_dst_matrix.shape[0]):
        intial_dst_matrix[i][i]=sys.maxsize
    
    #fitting data using complete linkage        
    clf=agglomerative(X)
    clf.fit(intial_dst_matrix)     
    
    #using complete linkage because it provides good clusters
    plt.figure(figsize =(8, 8))
    plt.title('Dendogram Average Linkage')
    shc.dendrogram((shc.linkage(X, method ='average')))
    plt.show()
    
   
    plt.title('Dendogram Singly linkage')
    shc.dendrogram((shc.linkage(X, method ='single')))
    plt.show()
    
    
    plt.title('Dendogram Complete Linkage ')
    shc.dendrogram((shc.linkage(X, method ='complete')))
    plt.show()
    
    colors= 10*["r","g","c","b","k"]
    
    plt.title('Agglomerative Clustering complete linkage cutting when clusters=4')    
    plt.xlabel('patient.age_at_initial_pathologic_diagnosis')
    plt.ylabel('patient.anatomic_neoplasm_subdivision')
    
   
    for i in clf.cluster:
        print("Data Points in Cluster ", i+1, " is")
        print(len(clf.cluster[i]))    
        for feat in clf.cluster[i]:
            color=colors[i]            
            plt.scatter(feat[0],feat[1], color = color,Label='True Position')
            
    plt.show() 
    fig = plt.figure()
    fig.suptitle('Agglomerative Clustering complete linkage cutting when clusters=4')
    
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('patient.age_at_initial_pathologic_diagnosis')
    ax.set_ylabel('patient.anatomic_neoplasm_subdivision')
    ax.set_zlabel('patient.axillary_lymph_node_stage_method_type')    
    for i in clf.cluster:
        color=colors[i]        
        for feat in clf.cluster[i]:          
            #plt.scatter(feat[0],feat[1],color=color,label='True Position')
            ax.scatter3D(feat[0],feat[1],feat[2], color = color,Label='True Position')
    
    plt.show()   
    
    
if __name__=="__main__":
    main()
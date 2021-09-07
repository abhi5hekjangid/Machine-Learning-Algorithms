# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:33:17 2021

@author: abhis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class k_means:
    def __init__(self,k,max_iter=200,tolerence=0.001):
         self.k=k
         self.tolerence=tolerence
         self.max_iter=max_iter
    #function to initialize intital clusters
    def initial_c_points(self,data):               
        self.centroids={}
        for i in range(self.k):
            self.centroids[i]=data[i]
        return self.centroids
    
    #function to compute within sum of square     
    def compute_wcss(self):
        for i in range(self.k):
            for point in self.classes[i]:
                self.inertia+=np.sum((point-self.centroids[i])**2)
        
        
    #to check if prev and curr centroids are same if yes then stop the iterations
    def compute_convergence(self,prev_centroids):
        for i in self.centroids:
                real_centroid=prev_centroids[i]                
                curr_centroid=self.centroids[i]                                
                if np.sum((curr_centroid-real_centroid)/real_centroid*100.0)> self.tolerence:
                    return False 
                
    def fit(self,data):       
        self.inertia=0.0
        #run for max iterations
        for i in range(self.max_iter):            
            #initializing first k points as initial centroids
            self.centroids=self.initial_c_points(data)            
            self.classes={}
            
            #using classes dictionary to store the clusters datapoints
            for i in range(self.k):
                self.classes[i]=[]
                
            #finding min ecludien distance point as new centroid and appending data into it                
            for feat in data:                                
                dst=[np.linalg.norm(feat-self.centroids[cent]) for cent in self.centroids]                                               
                near_centroid_index=dst.index(min(dst))               
                self.classes[near_centroid_index].append(feat)
                    
            #print("inside")
            #print(self.classes[i])
            #prev centroids to compare our new centroids
            prev_centroids=dict(self.centroids)            
            
            #finding mean of datapoints clusters to find new centroid
            for i in self.classes:
                self.centroids[i]=np.average(self.classes[i],axis=0)
            
            
            #checking if prev centroids is same as new centroids
            flag=self.compute_convergence(prev_centroids)              
            if flag:
                break       
        
        #within cluster sum of square computation
        self.compute_wcss()
    
                 
def main():
    df=pd.read_csv('BR_mod.csv')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(),inplace=True)
    X=np.array(df.iloc[:,:].astype(float))  
    
    #print(X.shape)
    """"df.plot(kind='scatter',x='patient.age_at_initial_pathologic_diagnosis'
            ,y='patient.anatomic_neoplasm_subdivisions.anatomic_neoplasm_subdivision'
            ,color='black')
    scaler=StandardScaler().fit(X)
    X_scaled=scaler.transform(X)"""
    
    #training for different values of k so that we can find optimal no of clusters
    dis=[]
    K=10
    for i in range(1,K+1):
         kmeanModel = k_means(k=i)
         kmeanModel.fit(X)         
         dis.append(kmeanModel.inertia)
   
    
    plt.plot(range(1,K+1), dis,'bx-')
    plt.xlabel('k')
    plt.ylabel('Within Cluster Sum of Square')
    plt.title('optimal k')
    plt.show()  
    
    classifier=k_means(k=4)
    classifier.fit(X)
    #print(classifier.inertia)
    
    colors= 10*["r","g","c","b","k"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('patient.age_at_initial_pathologic_diagnosis')
    ax.set_ylabel('patient.anatomic_neoplasm_subdivisions')
    ax.set_zlabel('patient.axillary_lymph_node_stage_method_type')
    #plotting clusters in 2d
    for i in classifier.centroids:        
        color=colors[i]       
        #plt.scatter(classifier.centroids[i][0],classifier.centroids[i][1],marker="*",color=color)        
        ax.scatter3D(classifier.centroids[i][0],classifier.centroids[i][1],classifier.centroids[i][2], linewidths=5,color = color,marker="x")
    
    for i in classifier.classes:
        color=colors[i]
        print("No of points in cluser ", i+1 ," is ", len(classifier.classes[i]))
        for feat in classifier.classes[i]:          
            #plt.scatter(feat[0],feat[1],color=color,label='True Position')
            ax.scatter3D(feat[0],feat[1],feat[2], color = color,Label='True Position')
    
    plt.show()        
    
    #plotting clusters in 3d   
    plt.xlabel('patient.age_at_initial_pathologic_diagnosis')
    plt.ylabel('patient.anatomic_neoplasm_subdivisions')
    for i in classifier.centroids:        
        color=colors[i]       
        plt.scatter(classifier.centroids[i][0],classifier.centroids[i][1],marker="*",color=color)  
    
    for i in classifier.classes:
        color=colors[i]
        for feat in classifier.classes[i]:          
            plt.scatter(feat[0],feat[1],color=color,label='True Position')       
                
    plt.show()        

if __name__== "__main__":
    main()

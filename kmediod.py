# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 23:45:46 2021

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
         
         
    #function to intialize medoids
    def initial_c_points(self,data):               
        self.medoids={}
        for i in range(self.k):
            self.medoids[i]=data[i]
        return self.medoids    
    
    #compairing cost of each iteration to previous iteration            
    # if cost of current and prev iterations are equal then we got out final cluster no need to run further
    def compare_costs(self,cost,new_cost):
        if cost>new_cost:
            self.medoids=self.new_medoids
            self.classes=self.new_classes
            self.cost=self.new_cost
            return True
        elif cost<=new_cost:
            return False
        
    #function to compute within sum of square     
    def compute_wcss(self):
        for i in range(self.k):
            for point in self.classes[i]:
                self.inertia+=np.sum((point-self.medoids[i])**2)
                
                
    def fit(self,data):       
        self.inertia=0.0
        self.cost=0.0
        #run for max iterations
        self.medoids=self.initial_c_points(data)
        self.classes={}
        
        for i in range(self.k):
                self.classes[i]=[]
        #appending each point to closet medoid
        for feat in data:                                
                dst=[np.linalg.norm(feat-self.medoids[cent]) for cent in self.medoids]                                               
                near_centroid_index=dst.index(min(dst))
                self.cost+=min(dst)
                self.classes[near_centroid_index].append(feat)
        
        for i in range(self.max_iter):          
            
            self.new_medoids={}           
            self.new_classes={}          
            self.new_cost=0.0
            
            for i in range(self.k):
                self.new_classes[i]=[]
            #finding distance of each point from all the point in same class and then new medoid of this class will
            #be the point whose distance is minmium compare to others
            for i in range(self.k):
                all_dst=[]
                for point in self.classes[i]:
                    dst=[np.linalg.norm(p-point) for p in self.classes[i]]                                               
                    all_dst.append(dst)
                #new mediod point inside a cluster    
                new_mediod_index=all_dst.index(min(all_dst))               
                self.new_medoids[i]=self.classes[i][new_mediod_index]
            
            #assigning points to new clusters using new medoids                
            for feat in data:                                
                dst=[np.linalg.norm(feat-self.new_medoids[point]) for point in self.new_medoids]                                               
                near_medoid_index=dst.index(min(dst))
                self.new_cost+=min(dst)
                self.new_classes[near_medoid_index].append(feat)            
            
            #compairing cost of each iteration to previous iteration            
            # if cost of current and prev iterations are equal then we got out final cluster no need to run further
            if self.compare_costs(self.cost,self.new_cost)==False:
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
    
    #training for different valus of k to find optimal k
    dis=[]
    K=10
    for i in range(1,K+1):
         kmeanModel = k_means(k=i)
         kmeanModel.fit(X)         
         dis.append(kmeanModel.inertia)
   
    
    plt.plot(range(1,K+1), dis,'bx-')
    plt.xlabel('k')
    plt.ylabel('Within Cluster Sum of Square')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()      
    
   
    
    classifier=k_means(k=4)
    classifier.fit(X)
    #print(classifier.inertia)
    
    colors= 10*["r","g","c","b","k"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('patient.age_at_initial_pathologic_diagnosis')
    ax.set_ylabel('patient.anatomic_neoplasm_subdivision')
    ax.set_zlabel('patient.axillary_lymph_node_stage_method_type')
    
    #plotting clusters in 3d
    for i in classifier.medoids:        
        color=colors[i]       
        #plt.scatter(classifier.medoids[i][0],classifier.medoids[i][1],marker="*",color=color,linewidths=5)        
        ax.scatter3D(classifier.medoids[i][0],classifier.medoids[i][1],classifier.medoids[i][2], color = color,marker="x",linewidths=5)
    
    for i in classifier.classes:
        color=colors[i]
        print("No of points in cluser ", i+1 ," is ", len(classifier.classes[i]))
        for feat in classifier.classes[i]:          
            #plt.scatter(feat[0],feat[1],color=color,label='True Position')
            ax.scatter3D(feat[0],feat[1],feat[2], color = color,Label='True Position')
    plt.show()   
    
    plt.xlabel('patient.age_at_initial_pathologic_diagnosis')
    plt.ylabel('patient.anatomic_neoplasm_subdivisions')
    
    #plotting clusters in 2d
    for i in classifier.medoids:        
        color=colors[i]       
        plt.scatter(classifier.medoids[i][0],classifier.medoids[i][1],marker="*",color=color,linewidths=5)
        
    for i in classifier.classes:
        color=colors[i]
        for feat in classifier.classes[i]:          
            plt.scatter(feat[0],feat[1],color=color,label='True Position')
    
    plt.show()           

if __name__== "__main__":
    main()

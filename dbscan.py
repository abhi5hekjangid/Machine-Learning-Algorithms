# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:08:15 2021

@author: abhis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
class dbscan:
    def __init__(self,X,ep,min_points):
        self.ep=ep
        self.min_points=min_points
        self.X=X
        print("Eps value is",self.ep)
        print("Min Points are",self.min_points)
    
    #fucntion to find the euclidean distance    
    def euclidean_dst(self,a,b):        
        dist = np.sqrt(np.sum(np.square(a-b)))        
        return dist
    
    #function to find neighbors of a point based on euclidean distance        
    def find_neighbors(self,point):
        nei=[]        
        for data in range(self.X.shape[0]):
            if data!=point:
                dst=self.euclidean_dst(self.X[data],self.X[point])                
                if dst<self.ep:
                    nei.append(data)
        return nei     
    #function to find cluster of a point
    def find_cluster(self,point,neighbors):
        cluster=[]
        cluster.append(point)
        
        #itertively finding core and border points
        for neighbor in neighbors:
            
            if neighbor not in self.visited:
                self.visited.append(neighbor)
                
                #finding core and border points
                self.neighbors[neighbor]=self.find_neighbors(neighbor)
                
                if len(self.neighbors[neighbor])>=self.min_points: 
                    #finding core points of a cluster                          
                    self.core[self.index].append(neighbor)
                    #find cluster for this core point
                    clus=self.find_cluster(neighbor,self.neighbors[neighbor])
                    cluster+=clus
                else:
                    cluster.append(neighbor)
                    #finding border point of a cluster
                    self.border[self.index].append(neighbor)
                    
        return cluster
   
    def fit(self):
        self.clusters=[]         
        self.visited=[]
        self.neighbors={}
        self.core={}
        self.border={}
        
        for point in range(self.X.shape[0]):
            
            #if the point is not visited             
            if point not in self.visited:
                
                #find neighbors of the unvisited point
                self.neighbors[point]=self.find_neighbors(point)
                
                #check for core point
                if len(self.neighbors[point])>=self.min_points:
                    self.visited.append(point) 
                    
                    #to find core and border points
                    self.core[point]=[]
                    self.border[point]=[]
                    self.index=point
                    
                    self.core[point].append(point)
                    #finding cluster of core point
                    cluster=self.find_cluster(point,self.neighbors[point])
                    self.clusters.append(cluster)
        #noise point in dataset            
        self.noise=[]
        for point in range(self.X.shape[0]):
            if point not in self.visited:
                self.noise.append(point)
        self.clusters.append(self.noise)     
def main():    
   data = arff.loadarff('diabetes1.arff')
   df = pd.DataFrame(data[0])
   df.replace([np.inf, -np.inf], np.nan, inplace=True)
   df.fillna(df.mean(),inplace=True)   
   df.drop('class', axis=1, inplace=True)
   
   X=np.array(df.iloc[:,1:].astype(float))
   
   scaler = StandardScaler()
   scaler.fit(X)
   X = scaler.transform(X)
   
   clu=dbscan(X,2,5)
   clu.fit()
   
  
   colors= ["r","g","c","b","k"] 

   
   i=1
   for index in clu.core:
       print("Plotting core and border points of cluster ",i," (core=red, border=blue)" )
       print("No of core points in cluster ", i ," is ", len(clu.core[index]))
       for point in clu.core[index]:
           plt.scatter(X[point][0],X[point][1],color="r")   
       
       print("No of border points in cluster ", i," are ", len(clu.border[index]))
       for point in clu.border[index]:
           plt.scatter(X[point][0],X[point][1],color="b")
       i+=1
       plt.show()
       
   #plotting noise points
   print("Noise Points are\nNo of Noise Points is ",len(clu.noise))
   for point in clu.noise:
       plt.scatter(X[point][0],X[point][1],color="k")
   plt.show()    
   
   print("\nCluster of whole dataset")
   #plotting all the clusters
   for i in range(len(clu.clusters)):
       color=colors[i]
       if i!=len(clu.clusters)-1:
           print("Data Points in cluster ",i+1," are ", len(clu.clusters[i]))
       
       for feat in clu.clusters[i]:
           plt.scatter(X[feat][0],X[feat][1],color=color,label='True Position')       
                
   plt.show()        
   
if __name__=="__main__":
    main()
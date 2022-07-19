
import matplotlib.pyplot as plt
import pandas as pd #导入pandas库
import numpy as np #导入numpy库
import basic
from scipy import stats
import os


class draw:
    Gapsize = basic.Gapsize
    Xsize = basic.Xsize
    Ysize = basic.Ysize
    Unitsize = basic.Unitsize
    Trainsize = basic.Trainsize
    dist = getattr(stats, 'norm')
    
    Squaresize = Xsize*Ysize
   
    def __init__(self,ip,op):    #初始化参数，基本评定中包括每个点的均值方差（二维）以及理应所在位置，然后还要有个点阵的总偏差
        self.ip = ip
        self.op = op
        
        self.ip_0 = ip[:,0]
        self.ip_1 = ip[:,1]
        self.ip_2 = ip[:,2]
        self.ip_3 = ip[:,3]
        
        self.op_x = op[:,0]
        self.op_y = op[:,1]
      
        
    def draw(self,path):
        isExists=os.path.exists(path+'\pic')
        if not isExists:
            os.makedirs(path+'\pic')
       
            
        map_a = dict()
        x = self.op_x
        y = self.op_y
        names = locals()
        
        for index in range(len(x)):
            if 'X'+str(x[index])+'Y'+str(y[index]) in map_a:
                names.get('X'+str(x[index])+'Y'+str(y[index]))[0].append(self.ip_0[index])
                names.get('X'+str(x[index])+'Y'+str(y[index]))[1].append(self.ip_1[index])
                names.get('X'+str(x[index])+'Y'+str(y[index]))[2].append(self.ip_2[index])
                names.get('X'+str(x[index])+'Y'+str(y[index]))[3].append(self.ip_3[index])
                
            else:
                map_a['X'+str(x[index])+'Y'+str(y[index])] = 1
                names['X'+str(x[index])+'Y'+str(y[index])] = [[] for i in range(4)]
        
    
     
        for index in map_a:
            
            plt.subplot(221)
            n, bins, patches = plt.hist(names.get(index)[0],bins=30)
            plt.subplot(222)
            n, bins, patches = plt.hist(names.get(index)[1],bins=30)
            plt.subplot(223)
            n, bins, patches = plt.hist(names.get(index)[2],bins=30)
            plt.subplot(224)
            n, bins, patches = plt.hist(names.get(index)[3],bins=30)
            
            plt.suptitle(index)
            plt.savefig(path+'/pic/'+index+'.jpg')
            plt.close()
            
        
        



        
                
                
                

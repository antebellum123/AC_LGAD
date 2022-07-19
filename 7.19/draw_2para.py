
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd #导入pandas库
import numpy as np #导入numpy库
import basic
from scipy import stats
import os

from scipy.stats import norm


class draw_2para:
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
        
        self.op_x = op[:,0]
        self.op_y = op[:,1]
      
        
    def draw(self,path):
        isExists=os.path.exists(path+'\pic2')
        if not isExists:
            os.makedirs(path+'\pic2')
       
        
        map_a = dict()
        x = self.op_x
        y = self.op_y
        names = locals()
        
        for index in range(len(x)):
            if 'X'+str(x[index]) in map_a:
                names.get('X'+str(x[index]))[0].append(self.ip_0[index])
                
            else:
                map_a['X'+str(x[index])] = 1
                names['X'+str(x[index])] = [[] for i in range(1)]
        
    
     
        for index in map_a:
            
            n, bins, patches = plt.hist(names.get(index)[0],bins=60,range=(-0.3,0.3),density=1)
            
            mu = np.mean(names.get(index)[0])
            sigma = np.std(names.get(index)[0])
             
            y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
            plt.plot(bins, y, 'r--') #绘制y的曲线 
            plt.suptitle(index+":mu="+str(mu)+" sigma="+str(sigma))
            plt.savefig(path+'/pic2/'+index+'.jpg')
            plt.close()
            
        for index in map_a:
            mu = np.mean(names.get(index)[0])
            sigma = np.std(names.get(index)[0])
             
            y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
            plt.plot(bins, y, 'r--') #绘制y的曲线 

        plt.savefig(path+'/pic2/'+'total.jpg')
        plt.close()
            
            
            
        
        



        
                
                
                

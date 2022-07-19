
import matplotlib.pyplot as plt
import pandas as pd #导入pandas库
import numpy as np #导入numpy库
import basic
from scipy import stats
import os


class judge:
    Xsize = basic.Xsize
    Ysize = basic.Ysize
    Unitsize = basic.Unitsize
    Trainsize = basic.Trainsize
    dist = getattr(stats, 'norm')
    
    Squaresize = Xsize*Ysize
    in_Squaresize = (Xsize-1)*(Ysize-1)
    
    
    def __init__(self,pos_test,pos_real):    #初始化参数，基本评定中包括每个点的均值方差（二维）以及理应所在位置，然后还要有个点阵的总偏差
        self.pos_test = pos_test
        self.pos_real = pos_real
        
        Xpos_wish = np.zeros((judge.Squaresize,1))
        Ypos_wish = np.zeros((judge.Squaresize,1))#二维
        Xpos_miu = np.zeros((judge.Squaresize,1))
        Ypos_miu = np.zeros((judge.Squaresize,1))
        Xpos_sigma = np.zeros((judge.Squaresize,1))
        Ypos_sigma = np.zeros((judge.Squaresize,1))
        Xbias = 0 
        Ybias = 0
        in_Xbias= 0
        in_Ybias= 0
        
        
        Test_Unitsize = int(len(pos_real)/judge.Squaresize)
      
        for i in range(0,judge.Squaresize):
            Xpos_wish[i][0] = self.pos_real[i][0]
            Ypos_wish[i][0]= self.pos_real[i][1]
            Xtemp = np.zeros(Test_Unitsize)
            Ytemp = np.zeros(Test_Unitsize)
            for  j in range(0,Test_Unitsize):
                Xtemp[j] = pos_test[i+judge.Squaresize*j][0]#存储顺序为先逐行扫描x轴点
                Ytemp[j] = pos_test[i+judge.Squaresize*j][1]
            
            temp = judge.dist.fit(Xtemp)
            Xpos_miu[i][0] = temp[0]
            Xpos_sigma[i][0] = temp[1]#此处重复运算两次，可以优化
            
            temp = judge.dist.fit(Ytemp)
            Ypos_miu[i][0] = temp[0]
            Ypos_sigma[i][0] = temp[1]
            
        self.matrix_real = np.hstack((Xpos_wish,Ypos_wish))
        self.matrix_miu = np.hstack((Xpos_miu,Ypos_miu))
        self.matrix_sigma = np.hstack((Xpos_sigma,Ypos_sigma))
        self.sigma = np.mean(self.matrix_sigma)
        
        for i in range(0,judge.Squaresize):
            Xbias+= (self.matrix_miu[i][0]-self.matrix_real[i][0])**2/judge.Squaresize
            Ybias+= (self.matrix_miu[i][1]-self.matrix_real[i][1])**2/judge.Squaresize
            
        if judge.Xsize>1 and judge.Ysize>1:
            for m in range(1,judge.Ysize-1):
                for n in range(1,judge.Xsize-1):
                    i = m*judge.Ysize+n
                    in_Xbias+= (self.matrix_miu[i][0]-self.matrix_real[i][0])**2/judge.in_Squaresize
                    in_Ybias+= (self.matrix_miu[i][1]-self.matrix_real[i][1])**2/judge.in_Squaresize
            
        self.xbias = Xbias**0.5
        self.ybias = Ybias**0.5
        self.bias = (Xbias+Ybias)**0.5
        self.in_xbias = in_Xbias**0.5
        self.in_ybias = in_Ybias**0.5
        self.in_bias = (in_Xbias+in_Ybias)**0.5
    
    def __str__(self):
        print("sigma =",self.sigma)
        print("xbias =",self.xbias)
        print("ybias =",self.ybias)
        print("bias =",self.bias)
        print("in_xbias =",self.in_xbias)
        print("in_ybias =",self.in_ybias)
        print("in_bias =",self.in_bias)
        return " "
    
    
    
    
    
    
    def save_file(self,path,filename):
    
        isExists=os.path.exists(path+'\output')
        if not isExists:
            os.makedirs(path+'\output')

            
            
        

        plt.scatter(self.matrix_miu[:,0],self.matrix_miu[:,1],s=10,c='red',marker = '.')#平均位置
        plt.scatter(self.matrix_real[:,0],self.matrix_real[:,1],s=10,c='blue',marker = '.')
        plt.title(filename)
        plt.savefig(path+'\output\\'+filename + '.png')
        plt.clf()
        
        matrix = np.hstack((self.matrix_real,self.matrix_miu,self.matrix_sigma))
        data1 = pd.DataFrame(matrix)
        data1.columns = ['X','Y','Xmiu','Ymiu','Xsigma','Ysigma']
        bias_data = np.array([[self.sigma,self.xbias,self.ybias,self.bias,self.in_xbias,self.in_ybias,self.in_bias]])
        data2 = pd.DataFrame(bias_data)
        data2.columns = ['sigma','xbias','ybias','bias','in_xbias','in_ybias','in_bias']
        
        writer = pd.ExcelWriter(path+'\output\\'+filename + '.xls',model ='w')		# 写入Excel文件
        data1.to_excel(writer, 'page_1', float_format='%.6f')		# ‘page_1’是写入excel的sheet名
        data2.to_excel(writer, 'page_2', float_format='%.6f')
        writer.save()

        writer.close()
    #其他的评定方案都是可选的，建议均采取输出数组的方式，而不是添加到类属性中
    
     
    



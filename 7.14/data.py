import pandas as pd #导入pandas库
import numpy as np 
import os
import basic
'''
文件名组成
'''
#s1_1 = r'C:\Users\Lenovo\Documents\daily\22.3.23\W7_1_C3_4A1_150V_3\X'
#s1_1 = r'C:\Users\Lenovo\Documents\daily\22.3.23\W5_4_D3_4A1_350V\X'
s1_1 = r'C:\Users\Lenovo\Documents\daily\22.3.23\W5_1_B3_4A1_350V_3\X'
s1_2 = '_Y'
s1_3 = r'\C'
s2 = '--w--0'
s3 = '.csv'#文件名

'''
导入可调参数
'''
Xsize = basic.Xsize
Ysize = basic.Ysize
Unitsize = basic.Unitsize
Gapsize = basic.Gapsize
Parasize = basic.Parasize
Trainsize = basic.Trainsize

Xbegin = basic.Xbegin
Ybegin = basic.Ybegin
Timegap = basic.Timegap
'''
谨慎修改部分
'''
Totalsize =Xsize*Ysize*Unitsize
relation_train=np.zeros((int(Unitsize*Trainsize)*Xsize*Ysize, Parasize))#尽量定义定长数组减小内存消耗，改这里的大小设定出错想想整型转换会发生什么
relation_test=np.zeros((Totalsize-int(Unitsize*Trainsize)*Xsize*Ysize, Parasize))

number_train = -1
number_test = -1

count = 0



for current_i in range(0,int(Unitsize*Trainsize)):
    for m in range(0,Ysize):
        for n in range(0,Xsize):#三个循环地位上等价，处理和保存数据都在三个循环内部
            number_train+=1
            
            
            #存放位置列
            
            relation_train[number_train][0] = Xbegin + Gapsize*n
            relation_train[number_train][1] = Ybegin + Gapsize*m
           
 
           
            #存放四个pad峰值
            
            for current_j in range(1,5):
                s = "%04d" % current_i
                str1 = s1_1+str(m)+s1_2+str(n)+s1_3+str(current_j)+s2+str(s)+s3
                data = np.loadtxt(open(str1,"rb"),delimiter=",",skiprows=5,usecols=[0,1]) 
                time = data[:,0]
                amp = data[:,1]
                basic = 0 
                interg = 0
                for basic_i in range(0,40):
                    basic= basic+amp[basic_i]/40
                    
                for interg_i in range(40,180):
                    interg+=(amp[interg_i]-basic)*Timegap
                    
                relation_train[number_train][current_j+1] = max(amp) - basic#从2开始存储
                relation_train[number_train][current_j+5] = interg
                
    if int(10*(current_i+1)/(int(Unitsize*Trainsize)))>count:
        count+=1
        if count == 10:
            print("wait:train data finished "+str(10*count)+"%")
            count=0
        else:
            print("wait:train data finished "+str(10*count)+"%")
        
    
   
#两个集合最好分开写，因为测试集参数可能会小于训练集，分开写运行速度没有区别   
for current_i in range(int(Unitsize*Trainsize),Unitsize):
    for m in range(0,Ysize):
        for n in range(0,Xsize):#三个循环地位上等价，处理和保存数据都在三个循环内部
            number_test+=1
            
            
            #存放位置列
            
            relation_test[number_test][0] = Xbegin + Gapsize*n
            relation_test[number_test][1] = Ybegin + Gapsize*m
 
            
            #存放四个pad峰值
            
            for current_j in range(1,5):
                s = "%04d" % current_i
                str1 = s1_1+str(m)+s1_2+str(n)+s1_3+str(current_j)+s2+str(s)+s3
                data = np.loadtxt(open(str1,"rb"),delimiter=",",skiprows=5,usecols=[0,1]) 
                time = data[:,0]
                amp = data[:,1]
                basic = 0 
                for basic_i in range(0,40):
                    basic= basic+amp[basic_i]/40
                    
                for interg_i in range(40,180):
                    interg+=(amp[interg_i]-basic)#不建议乘以时间间隔，因为会导致数值变小最后保存为6位浮点数变成0.而且乘以相同系数对后续没有帮助
                    
                relation_test[number_test][current_j+1] = max(amp) - basic#从2开始存储
                relation_test[number_test][current_j+5] = interg
                
    if int(10*(current_i+1)/(Unitsize-int(Unitsize*Trainsize)))>count:
        count+=1
        if count == 10:
            print("wait:test data finished "+str(10*count)+"%")
            count=0
        else:
            print("wait:test data finished "+str(10*count)+"%")
            
'''
#存储为excel文件,区分训练集和测试集
'''

data = pd.DataFrame(relation_train)
data.columns = ['X','Y','Q1','Q2','Q3','Q4','I1','I2','I3','I4']

writer = pd.ExcelWriter('preprocess_train.xls',model ='w')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.6f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()


data = pd.DataFrame(relation_test)
data.columns = ['X','Y','Q1','Q2','Q3','Q4','I1','I2','I3','I4']

writer = pd.ExcelWriter('preprocess_test.xls',model ='w')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.6f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()

import pandas as pd #导入pandas库
import numpy as np #导入numpy库
from sklearn.linear_model import LinearRegression #导入机器学习库中的线性回归模块
import matplotlib.pyplot as plt
import joblib
import basic
import os
from scipy import stats
from judge import judge
from draw import draw
from draw_2para import draw_2para
import shutil
import matplotlib.pyplot as plt

'''
导入模型
'''
curr_path = os.path.abspath('.')



'''
导入数据
'''
data=pd.read_excel('preprocess_train.xls')


data_posx=data['X'].values.reshape(-1,1)#注意数据类型是二维，shape查看应该是（n,1)
data_posy=data['Y'].values.reshape(-1,1)

data_q1=data['Q1'].values.reshape(-1,1)
data_q2=data['Q2'].values.reshape(-1,1)
data_q3=data['Q3'].values.reshape(-1,1)
data_q4=data['Q4'].values.reshape(-1,1)

data_q5=data['I1'].values.reshape(-1,1)
data_q6=data['I2'].values.reshape(-1,1)
data_q7=data['I3'].values.reshape(-1,1)
data_q8=data['I4'].values.reshape(-1,1)

'''
处理数据
'''
data_testy_lin = (data_q1+data_q2-data_q3-data_q4)/(data_q1+data_q2+data_q3+data_q4)
data_testx_lin = (data_q1+data_q3-data_q2-data_q4)/(data_q1+data_q2+data_q3+data_q4)

'''
数据合并和标记
'''
pos_real = np.hstack((data_posx,data_posy))

test_lin= np.hstack((data_testx_lin,data_testy_lin))
test_nlin = np.hstack((data_q1,data_q2,data_q3,data_q4))
test_nlinp = np.hstack((data_q1,data_q2,data_q3,data_q4,data_q5,data_q6,data_q7,data_q8))


isExists=os.path.exists(curr_path+'\test')
if isExists:
    shutil.rmtree(curr_path+'\\test')  
    os.mkdir(curr_path+'\\test') #清空数据
test_path = curr_path+'\\test'

a = draw(test_nlin,pos_real)
a.draw(test_path)

b = draw_2para(test_lin,pos_real)
b.draw(test_path)




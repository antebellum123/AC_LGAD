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
import shutil


'''
导入模型
'''
curr_path = os.path.abspath('.')

isExists=os.path.exists(curr_path+'\output')
if isExists:
    shutil.rmtree(curr_path+'\output')  
    os.mkdir(curr_path+'\output') #清空数据

lin=joblib.load(curr_path+'\\model\\lin.pkl')
nlin=joblib.load(curr_path+'\\model\\nlin.pkl')
#nlinp=joblib.load(curr_path+'\\model\\nlinp.pkl')
GBDT_4=joblib.load(curr_path+'\\model\\GBDT_4.pkl')
#GBDT_8=joblib.load(curr_path+'\\model\\GBDT_8.pkl')
#ridge = joblib.load(curr_path+'\\model\\ridge.pkl')
#GBR_4=joblib.load(curr_path+'\\model\\GBR_4.pkl')
MLP = joblib.load(curr_path+'\\model\\MLP.pkl')
#KNN = joblib.load(curr_path+'\\model\\KNN.pkl')
#SVM = joblib.load(curr_path+'\\model\\SVM.pkl')
#lasso = joblib.load(curr_path+'\\model\\lasso.pkl')
#CART = joblib.load(curr_path+'\\model\\CART.pkl')
#ETR = joblib.load(curr_path+'\\model\\ETR.pkl')
#RFR = joblib.load(curr_path+'\\model\\RFR.pkl')
Ada = joblib.load(curr_path+'\\model\\Ada.pkl')
#Bag = joblib.load(curr_path+'\\model\\Bag.pkl')
'''
导入数据
'''
data=pd.read_excel('preprocess_test.xls')


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


'''
重建位置
'''
pos_lin=lin.predict(test_lin)
pos_nlin = nlin.predict(test_nlin)
#pos_nlinp = nlinp.predict(test_nlinp)

pos_GBDT_4= GBDT_4.predict(test_nlin)
#pos_GBDT_8= GBDT_8.predict(test_nlinp)
#pos_GBR_4 = GBR_4.predict(test_nlin)
pos_MLP = MLP.predict(test_nlin)
#pos_ridge = ridge.predict(test_nlin)
#pos_KNN = KNN.predict(test_nlin)
#pos_SVM= SVM.predict(test_nlin)
#pos_lasso= lasso.predict(test_nlin)
#pos_CART= CART.predict(test_nlin)
#pos_ETR= ETR.predict(test_nlin)
#pos_RFR= RFR.predict(test_nlin)
pos_Ada= Ada.predict(test_nlin)
#pos_Bag= Bag.predict(test_nlin)
'''
#标定重建规范，两个重建评定
'''
class_lin = judge(pos_lin, pos_real)
class_nlin = judge(pos_nlin, pos_real)
#class_nlinp = judge(pos_nlinp, pos_real)
class_GBDT_4 = judge(pos_GBDT_4, pos_real)
#class_GBDT_8 = judge(pos_GBDT_8, pos_real)
#class_GBR_4 = judge(pos_GBR_4, pos_real)
class_MLP = judge(pos_MLP, pos_real)
#class_ridge = judge(pos_ridge, pos_real)
#class_KNN = judge(pos_KNN, pos_real)
#class_SVM = judge(pos_SVM, pos_real)
#class_lasso = judge(pos_lasso, pos_real)
#class_CART = judge(pos_CART, pos_real)
#class_ETR = judge(pos_ETR, pos_real)
#class_RFR = judge(pos_RFR, pos_real)
class_Ada = judge(pos_Ada, pos_real)
#class_Bag = judge(pos_Bag, pos_real)
'''
保存文件，均由judge文件中类函数操作，需要增加评定以及输出的话都应该在外部操作
'''
class_lin.save_file(curr_path,'class_lin')
class_nlin.save_file(curr_path,'class_nlin')
#class_nlinp.save_file(curr_path,'class_nlinp')
class_GBDT_4.save_file(curr_path,'class_GBDT_4')
#class_GBDT_8.save_file(curr_path,'class_GBDT_8')
#class_GBR_4.save_file(curr_path,'class_GBR_4')
class_MLP.save_file(curr_path,'class_MLP')
#class_ridge.save_file(curr_path,'class_ridge')
#class_KNN.save_file(curr_path,'class_KNN')
#class_SVM.save_file(curr_path,'class_SVM')
#class_lasso.save_file(curr_path,'class_lasso')
#class_CART.save_file(curr_path,'class_CART')
#class_ETR.save_file(curr_path,'class_ETR')
#class_RFR.save_file(curr_path,'class_RFR')
class_Ada.save_file(curr_path,'class_Ada')
#class_Bag.save_file(curr_path,'class_Bag')
'''
画图
'''


#重建模块主要功能
#1.读取文件部分：读取重建算法（从上一级得到），读取预处理后的数据
#2.重建位置，通过第一步的读取，便可以重建出某次打击的位置
#3.输出部分：寻找一种能标定重建结果好坏的规范输出文件
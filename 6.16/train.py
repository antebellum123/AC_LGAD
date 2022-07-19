import pandas as pd #导入pandas库
import numpy as np #导入numpy库
from sklearn.linear_model import LinearRegression #导入机器学习库中的线性回归模块
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import RidgeCV
import joblib
import os
import matplotlib.pyplot as plt

'''
创建输出文件
'''
curr_path = os.path.abspath('.')
isExists=os.path.exists(curr_path+'\model')
if not isExists:
    os.makedirs(curr_path+'\model')

'''
读取数据
'''
data=pd.read_excel('preprocess_train.xls')

data_posy=data['Y'].values.reshape(-1,1)
data_posx=data['X'].values.reshape(-1,1)

data_q1=data['Q1'].values.reshape(-1,1)
data_q2=data['Q2'].values.reshape(-1,1)
data_q3=data['Q3'].values.reshape(-1,1)
data_q4=data['Q4'].values.reshape(-1,1)
data_q5=data['I1'].values.reshape(-1,1)
data_q6=data['I2'].values.reshape(-1,1)
data_q7=data['I3'].values.reshape(-1,1)
data_q8=data['I4'].values.reshape(-1,1)


'''
数据操作
'''
data_trainx_lin = (data_q1+data_q2-data_q3-data_q4)/(data_q1+data_q2+data_q3+data_q4)
data_trainy_lin = (data_q1+data_q3-data_q2-data_q4)/(data_q1+data_q2+data_q3+data_q4)


'''
#数据合并
'''
pos_real = np.hstack((data_posx,data_posy))
para_lin = np.hstack((data_trainx_lin,data_trainy_lin))
para_nlin = np.hstack((data_q1,data_q2,data_q3,data_q4))
para_nlinp = np.hstack((data_q1,data_q2,data_q3,data_q4,data_q5,data_q6,data_q7,data_q8))
'''
#训练模型模块
'''

'''
#线性回归
lin=LinearRegression() #创建单元线性回归模型，参数默认
nlin=LinearRegression()#创建多元线性回归模型
nlinp=LinearRegression()

lin.fit(para_lin,pos_real)#这里采用了多输出多输入线性回归，其实是视为x,y和两个参数都有多元线性回归关系，实际运行中另外一个参数比重是0，如果觉得不喜欢的话可以改为两个单线性模型，输出模型也要是两个
nlin.fit(para_nlin,pos_real)
nlinp.fit(para_nlinp,pos_real)#多元迭代
'''
'''
#岭回归ridge
model = RidgeCV(alphas=[0.1, 1.0, 10.0],cv=10).fit(para_nlin,pos_real)
alpha = model.alpha_
ridge = Ridge(max_iter=10000, alpha=alpha)
ridge.fit(para_nlin,pos_real)

#lasso
model = MultiTaskLassoCV(cv=20).fit(para_nlin,pos_real)
alpha = model.alpha_
lasso = Lasso(max_iter=10000, alpha=alpha)
lasso.fit(para_nlin,pos_real)

#KNN
KNN = KNeighborsRegressor()
KNN.fit(para_nlin,pos_real)
#SVM
SVM1 = SVR()
SVM= MultiOutputRegressor(SVM1)
SVM.fit(para_nlin,pos_real)

#MLP
MLP = MLPRegressor(solver='adam',activation='relu', alpha=0.01,hidden_layer_sizes=(100, 50), max_iter=500)
MLP.fit(para_nlin,pos_real)

#CART
CART = DecisionTreeRegressor()
CART.fit(para_nlin,pos_real)
#ETR
ETR = ExtraTreeRegressor()
ETR.fit(para_nlin,pos_real)
#RFR
RFR = RandomForestRegressor()
RFR.fit(para_nlin,pos_real)
'''

#Ada
Ada1 = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=1000, learning_rate=0.1, random_state=123)
Ada= MultiOutputRegressor(Ada1)
Ada.fit(para_nlin,pos_real)


'''
#GBC
GBDT_4 = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
mult_GBDT_4 = MultiOutputRegressor(GBDT_4)
mult_GBDT_4.fit(para_nlin,pos_real)

GBDT_8 = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
mult_GBDT_8  = MultiOutputRegressor(GBDT_8)
mult_GBDT_8.fit(para_nlinp,pos_real)

#GBR

GBR_4 = GradientBoostingRegressor(n_estimators=1000, max_depth=3, min_samples_split=3, learning_rate=0.1)
mult_GBR_4 = MultiOutputRegressor(GBR_4)
mult_GBR_4.fit(para_nlin,pos_real)

#Bag
Bag = BaggingRegressor()
Bag.fit(para_nlin,pos_real)
'''

#导出模型模型


    
#joblib.dump(lin,curr_path+'\\model\\lin.pkl')#joblib读取的时候无法自动转译\，但第一个\却可以被转译，建议都加上\万无一失
#joblib.dump(nlin,curr_path+'\\model\\nlin.pkl')
#joblib.dump(nlinp,curr_path+'\\model\\nlinp.pkl')
#joblib.dump(ridge,curr_path+'\\model\\ridge.pkl')
#joblib.dump(KNN,curr_path+'\\model\\KNN.pkl')
#joblib.dump(SVM,curr_path+'\\model\\SVM.pkl')
#joblib.dump(lasso,curr_path+'\\model\\lasso.pkl')
#joblib.dump(MLP,curr_path+'\\model\\MLP.pkl')
#joblib.dump(CART,curr_path+'\\model\\CART.pkl')
#joblib.dump(ETR,curr_path+'\\model\\ETR.pkl')
#joblib.dump(RFR,curr_path+'\\model\\RFR.pkl')
joblib.dump(Ada,curr_path+'\\model\\Ada.pkl')
#joblib.dump(Bag,curr_path+'\\model\\Bag.pkl')
#joblib.dump(mult_GBR_4,curr_path+'\\model\\GBR_4.pkl')
#joblib.dump(mult_GBDT_4,curr_path+'\\model\\GBDT_4.pkl')
#joblib.dump(mult_GBDT_8,curr_path+'\\model\\GBDT_8.pkl')


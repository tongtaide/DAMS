

"""======================== Import ========================"""
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import math
import scipy.stats
from scipy import stats
from keras.models import load_model


    
"""===================== Valuestandard ====================="""
class Valuestandard:
    """
    Valuestandard用于数据标准化
    
    参数介绍
    ========
    skip_list：当Skip为True时才生效，表示不需要进行标准化的变量的序号列表，类型为list，默认为None
    
    onehot_list：当OneHot为True时才生效，表示需要进行独热编码的变量的序号列表，类型为list，默认为None
    
    categories_list：当OneHot为True时才生效，表示需要进行独热编码的变量的种类数列表，与onehot_list相互对应，类型为list，默认为None
    
    Skip：是否跳过部分变量，不对其进行标准化，如：二分类变量等，类型为bool，默认为False，即将全部变量标准化
    
    OneHot：是否使用独热编码，类型为bool，默认为False，即不使用独热编码
    ========
    """
    
    def __init__(self, skip_list=None, onehot_list=None, categories_list=None, Skip=False, OneHot=False):
        
        self.__Skip = Skip
        self.__OneHot = OneHot
        
        if self.__Skip:
            if type(skip_list) == type(None):
                raise ValueError("需要参数：skip_list")
            elif type(skip_list) != type([]):
                raise ValueError("参数“skip_list”的类型必须为：list")
            else:
                self.__list1 = skip_list
        
        if self.__OneHot:
            if type(onehot_list) == type(None):
                raise ValueError("需要参数：onehot_list")
            elif type(onehot_list) != type([]):
                raise ValueError("参数“onehot_list”的类型必须为：list")
            else:
                self.__list2 = onehot_list
        
        if self.__OneHot:
            if type(categories_list) == type(None):
                raise ValueError("需要参数：categories_list")
            elif type(categories_list) != type([]):
                raise ValueError("参数“categories_list”的类型必须为：list")
            else:
                for self.__k in range(len(categories_list)):
                    if self.__k == 0:
                        self.__list3 = [list(range(categories_list[self.__k]))]
                    else:
                        self.__list3.append(list(range(categories_list[self.__k])))
                
        self.__fitstate = False
        self.__fitstate_2 = False
        
        return None
    
    def fit_transform(self, data_1=None):
        """
        fit_transform 方法用于标准化数据
        
        参数介绍
        ========
        data_1：待标准化的数据集，类型为DataFrame，默认为None
        ========
        """
        
        if self.__fitstate:
            raise RuntimeError("数据已被标准化，请勿重复标准化；若要重新标准化，请重新实例化Valuestandard")
        
        if type(data_1) == None:
            raise ValueError("需要待标准化的数据集：data_1")
        elif type(data_1) != type(pd.DataFrame()):
            raise ValueError("数据集的类型必须为：DataFrame，而传入数据的类型为：%s" % type(data_1))
        else:
            self.__Data_1 = data_1.copy()
            self.__data = self.__Data_1.copy()
            
        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)
            
            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)
                
                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)
                
                self.__transdata = None
                self.__quantity = None
                self.__OH = None
                self.__LE = None
                
            self.__Oname_list = None
            self.__data_1 = None
        
        if self.__Skip:
            self.__Sname_list = self.__Data_1.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            if self.__data.shape[1] != 0:
                self.__s = True
            else:
                self.__s = False
            
            self.__Sname_list = None
            
        if self.__s:
            self.__Dname_list = self.__data.columns
            self.__data_original = self.__data.copy()
            self.__Key = StandardScaler()
            self.__data_standard = self.__Key.fit_transform(self.__data)
            self.__data_standard = pd.DataFrame(self.__data_standard)
            self.__data_standard.columns = self.__Dname_list
            self.__Dname_list = None
        else:
            self.__data_standard = self.__data.copy()
            self.__data_original = self.__data.copy()
        
        if self.__Skip:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_skip], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_o2], axis=1)
            self.__data_skip = None
            self.__data_o2 = None
        
        if self.__OneHot:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_onehot], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_onehot], axis=1)
            self.__data_onehot = None
        
        self.__data = None
        self.__fitstate = True
        
        return None
    
    def transform(self, data_2=None):
        """
        transform 方法用于，以fit_transform的标准，标准化数据，如测试集数据等
        
        参数介绍
        ========
        data_2：待标准化的数据集，类型为DataFrame，默认为None
        
        ★可以调用多次，但只保存最后一次调用的数据★
        ========
        """
        if self.__fitstate != True:
            raise RuntimeError("缺少数据标准化标准，请先调用 fit_transform() 方法")
        
        if type(data_2) == None:
            raise ValueError("需要待标准化的数据集：data_2")
        elif type(data_2) != type(pd.DataFrame()):
            raise ValueError("数据集的类型必须为：DataFrame，而传入数据的类型为：%s" % type(data_2))
        else:
            self.__Data_2 = data_2.copy()
            self.__data = self.__Data_2.copy()
            
        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)
            
            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)
                
                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)
                
                self.__transdata = None
                self.__quantity = None
                self.__OH = None
                self.__LE = None
                
            self.__Oname_list = None
            self.__data_1 = None
        
        if self.__Skip:
            self.__Sname_list = self.__Data_2.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            if self.__data.shape[1] != 0:
                self.__st = True
            else:
                self.__st = False
            
            self.__Sname_list = None
            
        if self.__st:
            self.__Dname_list = self.__data.columns
            self.__data_original_2 = self.__data.copy()
            self.__data_standard_2 = self.__Key.transform(self.__data)
            self.__data_standard_2 = pd.DataFrame(self.__data_standard_2)
            self.__data_standard_2.columns = self.__Dname_list
            self.__Dname_list = None
        else:
            self.__data_standard_2 = self.__data.copy()
            self.__data_original_2 = self.__data.copy()
        
        if self.__Skip:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_skip], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_o2], axis=1)
            self.__data_skip = None
            self.__data_o2 = None
        
        if self.__OneHot:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_onehot], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_onehot], axis=1)
            self.__data_onehot = None
        
        self.__data = None
        self.__fitstate_2 = True
        
        return None
        
    
    def standard_data_1(self):
        """
        standard_data_1 用于返回fit_transform()方法标准化后的数据，需先调用fit_transform()方法
        """
        if self.__fitstate != True:
            raise RuntimeError("没有数据被标准化，请先调用 fit_transform() 方法")
        
        return self.__data_standard
    
    def standard_data_2(self):
        """
        standard_data_2 用于返回transform()方法标准化后的数据，需先调用transform()方法
        """
        if self.__fitstate_2 != True:
            raise RuntimeError("没有数据被标准化，请先调用 transform() 方法")
        
        return self.__data_standard_2
    
    def original_data_1(self):
        """
        用于返回fit_transform()方法标准化后的原始数据，需先调用fit_transform()方法
        """
        if self.__fitstate != True:
            raise RuntimeError("请先调用 fit_transform() 方法")
        
        return self.__data_original
    
    def original_data_2(self):
        """
        用于返回transform方法标准化后的原始数据，需先调用transform方法
        """
        if self.__fitstate_2 != True:
            raise RuntimeError("请先调用 transform() 方法")
            
        return self.__data_original_2

"""========================================================="""



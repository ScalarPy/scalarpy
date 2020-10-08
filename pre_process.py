import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import math
from IPython.display import display
import random
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn.preprocessing as sk
import sklearn.model_selection as skm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import datefinder



class preprocess:
    def __init__(self,dataset,target=None,cat_thresh=20,ignore_columns=None):
        self.data=dataset.copy()
        self.dataset_cat=pd.DataFrame()
        self.dataset_num=pd.DataFrame()
        self.dataset_datetime=pd.DataFrame()
        self.dataset_high_cardinality=pd.DataFrame()
        self.target=target
        self.ignore_columns=ignore_columns
        self.col_i=self.data.columns
        self.cat_thresh=cat_thresh;
        if(ignore_columns):
            self.data=self.data.drop(ignore_columns,axis=1)
        self.data=self.data.replace([np.inf,-np.inf],np.NaN)
        self.col_ni=self.data.columns
        
        for col in self.data.columns:
            if self.data[col].dtype=="object":
                try:
                    con1=dataset[col].astype("str").str.match("^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$").any()
                    con2=dataset[col].astype("str").str.match('(\d{4})-(\d{2})-(\d{2})( (\d{2}):(\d{2}):(\d{2}))?').any()
                    con3=dataset[col].astype("str").str.match('^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|(?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$').any()
                    con4=dataset[col].astype("str").str.match('^([0-9]|0[0-9]|1[0-9]|2[0-3]):([0-9]|[0-5][0-9]:)?([0-5]?\d)$').any()
                    m=datefinder.find_dates(dataset[dataset[col].notnull()][col].astype("str")[0])
                    if(con1 or con2 or con3 or con4 or len(list(m))):
                        self.data[col]=pd.to_datetime(self.data[col],errors="coerce")
                        #self.dataset_datetime
                        #Generate DateTime Features
                        time_fe=pd.DataFrame()
                        if(len(self.data[col].dt.year.value_counts())>1):
                            time_fe[col+"_year"]=self.data[col].dt.year
                        if(len(self.data[col].dt.month.value_counts())>1):
                            time_fe[col+"_month"]=self.data[col].dt.month
                        if(len(self.data[col].dt.day.value_counts())>1):
                            time_fe[col+"_day"]=self.data[col].dt.day
                        if(len(self.data[col].dt.hour.value_counts())>1):
                            time_fe[col+"_hour"]=self.data[col].dt.hour
                        if(len(self.data[col].dt.minute.value_counts())>1):
                            time_fe[col+"_minute"]=self.data[col].dt.minute
                        if(len(self.data[col].dt.second.value_counts())>1):
                            time_fe[col+"_second"]=self.data[col].dt.second
                        if(len(self.data[col].dt.dayofweek.value_counts())>1):
                            time_fe[col+"_dayofweek"]=self.data[col].dt.dayofweek
                        #print(self.data[col])
                        self.data=self.data.drop(col,axis=1)
                        self.data=pd.concat([self.data,time_fe],axis=1)
                except:
                    continue
        #display(self.data)
        #display(self.data.dtypes)
        for col in self.data.columns:
            if(self.data[col].nunique()<cat_thresh):
                 self.dataset_cat[col]=self.data[col]
            elif(self.data[col].dtype=='object' and self.data[col].nunique()>cat_thresh):
                 self.dataset_high_cardinality[col]=self.data[col]
            elif((self.data[col].dtype=='int64' or self.data[col].dtype=='float64') and self.data[col].nunique()>cat_thresh):
                  self.dataset_num[col]=self.data[col]
        
    def impute_missing(self,numerical_imputation="mean",categorical_imputation="mode"):
        dataset_high_cardinality=pd.DataFrame()
        dataset_cat=pd.DataFrame()
        if(numerical_imputation=="mean"):
            dataset_num= self.dataset_num.fillna(self.dataset_num.mean())
        elif(numerical_imputation=="median"):
            dataset_num= self.dataset_num.fillna(self.dataset_num.median())
        elif(numerical_imputation=="mode"):
            dataset_num= self.dataset_num.fillna(self.dataset_num.mode().iloc[0,:])
        if(categorical_imputation=="mode"):
            if(not self.dataset_cat.empty):
                dataset_cat= self.dataset_cat.fillna(self.dataset_cat.mode().iloc[0,:])
            if(not self.dataset_high_cardinality.empty):
                dataset_high_cardinality= self.dataset_high_cardinality.fillna(self.dataset_high_cardinality.mode().iloc[0,:])
        self.data=pd.concat([dataset_num,dataset_cat,dataset_high_cardinality],axis=1)
        return  self.data
    def handle_outliers(self,method="iqr",outlier_threshold=2,strategy="replace_lb_ub",columns="all"):
        if(method=="iqr"):
            if columns=="all":
                for col in  self.dataset_num:
                    q1= self.data[col].describe()["25%"]
                    q3= self.data[col].describe()["75%"]
                    iqr=q3-q1
                    lb=q1-(1.5*iqr)
                    ub=q3+(1.5*iqr)
                    out= self.data[( self.data[col]<lb) | ( self.data[col]>ub)]
                    num_o=out.shape[0]
                    p=(num_o/self.data.shape[0])*100
                    if(p<outlier_threshold and p>0):
                        if(strategy=="replace_lb_ub"):
                            outlier_dict={}.fromkeys( self.data[ self.data[col]>ub][col],ub)
                            outlier_dict.update({}.fromkeys( self.data[ self.data[col]<lb][col],lb))
                            self.data[col]= self.data[col].replace(outlier_dict)
                        elif(strategy=="replace_mean"):
                            outlier_dict_mean={}.fromkeys( self.data[( self.data[col]<lb) | ( self.data[col]>ub)][col], self.data[col].mean())
                            self.data[col]= self.data[col].replace(outlier_dict_mean)
                        elif(strategy=="replace_median"):
                            outlier_dict_median={}.fromkeys( self.data[( self.data[col]<lb) | ( self.data[col]>ub)][col], self.data[col].median())
                            self.data[col]= self.data[col].replace(outlier_dict_median)  
                        elif(strategy=="remove"):
                            #outlier_index=data[(data[col]<lb) | (data[col]>ub)].index
                            #print()
                            self.data= self.data[( self.data[col]>lb) & (self.data[col]<ub)]
            else:
                for col in columns:
                    if(col in  self.dataset_num):
                        q1= self.data[col].describe()["25%"]
                        q3= self.data[col].describe()["75%"]
                        iqr=q3-q1
                        lb=q1-(1.5*iqr)
                        ub=q3+(1.5*iqr)
                        out= self.data[(self.data[col]<lb) | (self.data[col]>ub)]
                        num_o=out.shape[0]
                        p=(num_o/self.data.shape[0])*100
                        if(p<outlier_threshold and p>0):
                            if(strategy=="replace_lb_ub"):
                                outlier_dict={}.fromkeys( self.data[ self.data[col]>ub][col],ub)
                                outlier_dict.update({}.fromkeys( self.data[ self.data[col]<lb][col],lb))
                                self.data[col]= self.data[col].replace(outlier_dict)
                            elif(strategy=="replace_mean"):
                                outlier_dict_mean={}.fromkeys( self.data[( self.data[col]<lb) | ( self.data[col]>ub)][col], self.data[col].mean())
                                self.data[col]= self.data[col].replace(outlier_dict_mean)
                            elif(strategy=="replace_median"):
                                outlier_dict_median={}.fromkeys( self.data[( self.data[col]<lb) | ( self.data[col]>ub)][col], self.data[col].median())
                                self.data[col]= self.data[col].replace(outlier_dict_median)  
                            elif(strategy=="remove"):
                                #outlier_index=data[(data[col]<lb) | (data[col]>ub)].index
                                #print()
                                self.data= self.data[(self.data[col]>lb) & (self.data[col]<ub)]
        return  self.data
    def encode_data(self,strategy="one_hot_encode",high_cardinality="frequency",drop_first=True,ordinal_map=None,categorical_features="auto",encode_map=None):
        data= self.data
        target_encode=None
        data=self.impute_missing()
        #print(categorical_features) 
        self.categorical_features=categorical_features
        self.high_cardinality=high_cardinality
        if(self.target):
            if(data[self.target].dtype=="object"):
                label_encoder = LabelEncoder()
                data[self.target]=label_encoder.fit_transform(data[self.target])
            target_encode=data[self.target] 
            data=data.drop(self.target,axis=1)
        if(self.categorical_features=="auto"):
            self.categorical_features=[]
            self.high_cardinality_features=[]
            for col in data.columns:
                if(data[col].dtype=="object" and data[col].nunique()<self.cat_thresh):
                    self.categorical_features.append(col)
                elif(data[col].dtype=="object" and data[col].nunique()>self.cat_thresh):
                    self.high_cardinality_features.append(col)
        
        if(self.high_cardinality=="frequency"):
            self.hc_frequency_map={}
            for col in self.high_cardinality_features:
                self.hc_frequency_map[col]=dict(data[col].value_counts())
                data[col]=data[col].map(self.hc_frequency_map[col])
        if strategy=="one_hot_encode":
            self.oh_map={}
            
            for col in self.categorical_features:
                self.oh_map[col]=OneHotEncoder()
                oh_encode=pd.DataFrame(self.oh_map[col].fit_transform(data[col].values.reshape(-1,1)).toarray(),columns=[col+"_"+s for s in sorted(data[col].unique())])
                data=data.drop(col,axis=1)
                
                if(drop_first):
                    oh_encode=oh_encode.iloc[:,1:]
                data=pd.concat([data,oh_encode],axis=1)
        elif strategy=="label_encode":
            self.lb_map={}
            for col in self.categorical_features:
                self.lb_map[col] = LabelEncoder()
                data[col]=self.lb_map[col].fit_transform(data[col])
        elif strategy=="ordinal_encode":
            if not ordinal_map:
                raise ValueError("ordinal_map should not be None for Ordinal Encoding")
            else:
                for key,value in ordinal_map.items():
                    #num=list(range(0,len(value)))
                    #map_d = {value[i]: num[i] for i in range(len(value))} 
                    data[key]=data[key].map(value)    
        elif strategy=="frequency" or strategy=="count":
            self.frequency_map={}
            for col in self.categorical_features:
                self.frequency_map[col]=dict(data[col].value_counts())
                data[col]=data[col].map(self.frequency_map[col])
        elif strategy=="hybrid":
            if not encode_map:
                raise ValueError("encode_map should not be None for Hybrid Encoding")
            else:
                for key,value in encode_map.items():
                    if(key=="one_hot_encode"):
                        data=pd.get_dummies(data,columns=value,drop_first=drop_first)
                    elif(key=="label_encode"):
                        for col in value:
                            label_encoder = LabelEncoder()
                            data[col]=label_encoder.fit_transform(data[col])
                    elif(key=="ordinal_encode"):
                        for k,v in value.items():
                            num=list(range(0,len(v)))
                            map_d = {v[i]: num[i] for i in range(len(v))} 
                            data[k]=data[k].map(map_d)
        if(self.target):
            data=pd.concat([data,target_encode],axis=1)
        
        self.data=data
                                     
        return self.data    
    def normalize(self,method="min_max"):
        data=self.data
         
        target=None
        if(self.target):
            target=data[self.target]
            data=data.drop(self.target,axis=1)
            
        dataset_num=pd.DataFrame()
        dataset_cat=pd.DataFrame()
        for col in data.columns:
            if((data[col].dtype=='int64' or data[col].dtype=='float64')):
                dataset_num[col]=data[col]
            else:
                dataset_cat[col]=data[col]
        
        if(method=="min_max"):
            self.sc=sk.MinMaxScaler()
            col=dataset_num.columns
            dataset_num=pd.DataFrame(self.sc.fit_transform(dataset_num),columns=col)
        elif(method=="standard"):
            self.sc=sk.StandardScaler()
            col=dataset_num.columns
            dataset_num=pd.DataFrame(self.sc.fit_transform(dataset_num),columns=col)
        elif(method=="robust"):
            self.sc=sk.RobustScaler()
            col=dataset_num.columns
            dataset_num=pd.DataFrame(self.sc.fit_transform(dataset_num),columns=col)
        
        if(self.target):
            data=pd.concat([dataset_num,dataset_cat,target],axis=1)
        else:
            data=pd.concat([dataset_num,dataset_cat],axis=1)
        
        self.data=data
        return self.data
    
    def preprocess_data(self,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    normalization_strategy="min_max",verbose=1
                    ):
        print("Part-1 Data PreProcessing Started...")
        print(10*"=")
        self.missing=impute_missing
        self.outliers=handle_outliers
        self.encode=encode_data
        self.scale=normalize
        self.numerical_imputation=numerical_imputation
        self.categorical_imputation=categorical_imputation
        self.encode_strategy=encoding_strategy
        self.drop_first=encode_drop_first
        self.ordinal_map=ordinal_map
        self.categorical_features=encoding_categorical_features
        self.encode_map=encode_map
        if impute_missing:
            if(verbose):
                print("Handling Missing Values")
            self.impute_missing(numerical_imputation,categorical_imputation)
            if(verbose):
                print(30*"=")
        if handle_outliers:
            if(verbose):
                print("Handling Outliers Values")
            self.handle_outliers(outlier_method,outlier_threshold,outlier_strategy,outlier_columns)
            if(verbose):
                print(30*"=")
        if encode_data:
            if(verbose):
                print("Encoding Data")
            self.encode_data(encoding_strategy,high_cardinality_encoding,encode_drop_first,ordinal_map,encoding_categorical_features,encode_map)
            if(verbose):
                print(30*"=")
        if normalize:
            if(verbose):
                print("Normaliziling Values")
            self.normalize(normalization_strategy)
            if(verbose):
                print(30*"=")
        self.p_columns=self.data.columns
        return self.data
    
    def preprocess_data_new(self,new_data):
        
        if(type(new_data)==list):
            new_data=[new_data]
            if(len(new_data[0])==(len(self.col_i)-1)):
                col_d=list(self.col_i)
                col_d.remove(self.target)
                new_data=pd.DataFrame(new_data,columns=col_d)
            elif(len(new_data[0])==(len(self.col_ni)-1)):
                col_d=list(self.col_ni)
                col_d.remove(self.target)
                new_data=pd.DataFrame(new_data,columns=col_d)
            else:
                raise ValueError("Wrong Shape ("+str(len(new_data[0]))+",)")
        if(type(new_data)==pd.core.frame.DataFrame):
            for col in new_data:
                if(self.ignore_columns):
                    if(col in self.ignore_columns):
                        new_data=new_data.drop(col,axis=1)
        if self.missing:
            if(self.numerical_imputation=="mean"):
                new_data= new_data.fillna(new_data.mean())
            elif(self.numerical_imputation=="median"):
                new_data= new_data.fillna(new_data.median())
            elif(self.numerical_imputation=="mode"):
                new_data= new_data.fillna(new_data.mode().iloc[0,:])
            if(self.categorical_imputation=="mode"):
                new_data= new_data.fillna(new_data.mode().iloc[0,:])
        
        if self.encode:
            if(self.high_cardinality=="frequency"):
                for col in self.high_cardinality_features:
                    new_data[col]=new_data[col].map(self.hc_frequency_map[col])      
            if self.encode_strategy=="one_hot_encode":
                for col in self.categorical_features:
                    oh_encode=pd.DataFrame(self.oh_map[col].transform(new_data[col].values.reshape(-1,1)).toarray())
                    if self.drop_first:
                        oh_encode=oh_encode.iloc[:,1:]
                    new_data=new_data.drop(col,axis=1)
                    new_data=pd.concat([new_data,oh_encode],axis=1)
            elif self.encode_strategy=="label_encode":
                for col in self.categorical_features:
                    new_data[col]=self.lb_map[col].transform(new_data[col])
            elif self.encode_strategy=="ordinal_encode":
                if not self.ordinal_map:
                    raise ValueError("ordinal_map should not be None for Ordinal Encoding")
                else:
                    for key,value in self.ordinal_map.items():
                        num=list(range(0,len(value)))
                        map_d = {value[i]: num[i] for i in range(len(value))} 
                        new_data[key]=new_data[key].map(map_d)    
            elif self.encode_strategy=="frequency" or self.encode_strategy=="count":
                for col in self.categorical_features:
                    new_data[col]=new_data[col].map(self.frequency_map[col])                        
        if self.scale:
            new_data=pd.DataFrame(self.sc.transform(new_data))
        p_col=list(self.p_columns)
        p_col.remove(self.target)
        new_data.columns=p_col
        return new_data
    def handle_imbalance(self,x_train,y_train,resampling_method="smote",verbose=1):
        if(verbose):
                print("Resampling Data")
        if(resampling_method=="smote"):
            smo=SMOTE(k_neighbors=5)
            x_som,y_som=smo.fit_sample(x_train,y_train)
            return x_som,y_som
        elif(resampling_method=="over_sampler"):
            ros=RandomOverSampler()
            x_ros,y_ros=ros.fit_sample(x_train,y_train)
            return x_ros,y_ros
        elif(resampling_method=="under_sampler"):
            rus=RandomUnderSampler()
            x_rus,y_rus=rus.fit_sample(x_train,y_train)
            return x_rus,y_rus


        
        
      
        
        
        
            
        
        
        





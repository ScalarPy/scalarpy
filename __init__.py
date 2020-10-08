import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import math
from IPython.display import display
import random
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as sk
from scalarpy.pre_process import * 
from scalarpy.scalarpy_classifier import * 
from scalarpy.scalarpy_regressor import *
from ipywidgets import GridspecLayout
import ipywidgets as widgets
import pickle
import functools

#Read Datatset
def read_dataset(path=None,sql=None,con=None):
    """
    read_dataset function reads various kinds of file or sql query into a DataFrame, If path is specified reads a file into a DataFrame.If sql is specified reads SQL query or database table into a DataFrame.
    
    Supported File Formats-- csv, tsv, xlsx, json, h5
        
    Parameters
    ----------
    path : str , path object or file-like object 
            Any valid string path is acceptable. The string could be a URL. Valid
            URL schemes include http, ftp, s3, and file. For file URLs, a host is
            expected.
    
    sql : string or SQLAlchemy Selectable (select or text object)
            SQL query to be executed or a table name.
     
    con : SQLAlchemy connectable (engine/connection) or database string URI
        	or DBAPI2 connection (fallback mode)
    Using SQLAlchemy makes it possible to use any DB supported by that
       	 library. If a DBAPI2 object, only sqlite3 is supported.
    
    Returns
        
    -------
    DataFrame

    
    """
    if(path):
        file_type=os.path.splitext(path)[1]
        #print(file_type)
        data=None
        if (file_type==".csv"):
            data=pd.read_csv(path)
        elif (file_type==".tsv"):
            data=pd.read_csv(path,delimiter='\t')
        elif (file_type=='.xlsx'):
            data=pd.read_excel(path)
        elif (file_type=='.json'):
            data=pd.read_json(path)
        elif (file_type=='.h5'):
            data=pd.read_hdf(path)
        else:
            print("File type not supported")
    elif(sql):
        if(con):
            data=pd.read_sql(sql,con)
        else:
            raise ValueError("Connection object should not be None")
    else:
        raise ValueError("Either specify the path of the file or Sql query")
    return data

#Categorical Plots
def cat_plot(dataset,n=None,m=None,fig_size=(15,15),kind="bar",target=None,cat_thresh=20):
    """
    cat_plot(Categorical Plot) function automatically identifies the categorical variables in the given 
    data  and creates subplots. The default subplot if the target is not specified is the bar chart. 
    If target is specified the subplot depends on the target type. Support categorical plots pie,strip

    Parameters
    ----------
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    n : int, default=None
    Number of rows for subplot. If None, the value is computed based on the data size.
    
    m : int, default=None
    Number of columns for subplot.If None, the value is computed based on the data size.
        
    fig_size=tuple,	 default = (15,15)
    Represents Width and Height of the figure.
        
    kind=string, default=bar
    Type of visualization plot to be passed in as a string, Supported plots : pie,strip
    
    target:string, default=None
    Name of the target column to be passed in as a string. The target variable could be continuous, binary or  multiclass
    
    cat_thresh:int, default=10
    Threshold value which represents no of categories for identifying a categorical variable.
    
    ----------
    
    Returns
    ----------
    Visual Plot: Prints the visual plot.

    
    """
    dataset_cat=pd.DataFrame()
    for col in dataset.columns:
        if(dataset[col].nunique()<cat_thresh):
            dataset_cat[col]=dataset[col]
    if not n and not m:
        n=math.ceil(dataset_cat.shape[1]/2)
        m=math.ceil(dataset_cat.shape[1]/2)
    plt.figure(figsize=fig_size)
    for i,col in enumerate(dataset_cat,1):
        #print(i)
        plt.subplot(n,m,i)
        if target:
            if(dataset[target].nunique()<=10):
                sns.countplot(dataset_cat[col],hue=dataset[target])
            else:
                if(kind=="strip"):
                    sns.stripplot(dataset_cat[col],dataset[target])
                elif kind=="box":
                    sns.boxplot(dataset_cat[col],dataset[target])
                else:
                    sns.violinplot(dataset_cat[col],dataset[target])
        elif kind =="bar":
            sns.countplot(dataset_cat[col])
        elif kind =="pie":
            plt.title(col)
            fdt=dataset_cat[col].value_counts()
            plt.pie(fdt, labels=fdt.index, autopct='%1.1f%%',shadow=True, startangle=90)


#Numerical Plots
def num_plot(dataset,n=None,m=None,fig_size=(15,15),kind="hist",target=None):
    """
    num_plot(Numerical Plot) function automatically identifies the numerical variables in the given data and creates subplots. The default subplot if the target is not specified is the hist (Histogram). If target is specified the subplot depends on the target type. Support numerical plots box (Box Plot), dist (Distribution). 
    
    Parameters
    ----------
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    n : int, default=None
    Number of rows for subplot. If None, the value is computed based on the data size.
    
    m : int, default=None
    Number of columns for subplot.If None, the value is computed based on the data size.
        
    fig_size=tuple,	 default=(15,15)
    Represents Width and Height of the figure.
        
    kind=string, default=hist
    Type of visualization plot, to be passed in as a string, Supported plots : box, dist.
    
    target:string, default=None
    Name of the target column to be passed in as a string. The target variable could be continuous, binary or  multiclass.
    
    ----------
    
    Returns
    ---------
    Visual Plot: Prints the visual plot.

    
    """
    dataset_num=pd.DataFrame()
    for col in dataset.columns:
        if((dataset[col].dtype=='int64' or dataset[col].dtype=='float64') and dataset[col].nunique()>10):
            dataset_num[col]=dataset[col]
    if not n and not m:
        n=math.ceil(dataset_num.shape[1]/2)
        m=math.ceil(dataset_num.shape[1]/2)
    plt.figure(figsize=fig_size)
    for i,col in enumerate(dataset_num,1):
        #print(i)
        plt.subplot(n,m,i)
        if target:
            if(dataset[target].nunique()<=10):
                if(kind=="strip"):
                    sns.stripplot(dataset[target],dataset_num[col])
                elif kind=="box":
                    sns.boxplot(dataset[target],dataset_num[col])
                else:
                    sns.violinplot(dataset[target],dataset_num[col])
            else:
                plt.scatter(dataset[col],dataset[target])
                plt.xlabel(col)
                plt.ylabel(target)
        elif kind =="hist":
            plt.hist(dataset_num[col])
            plt.xlabel(col)
            plt.ylabel("Frequency")
        elif kind =="dist":
            dataset_num[col]=dataset_num[col].fillna(dataset_num[col].mean())
            sns.distplot(dataset_num[col])
            plt.xlabel(col)
        elif kind =="box":
            sns.boxplot(dataset_num[col])
            plt.xlabel(col)
            plt.ylabel("Frequency")
            

class clt:
    def central_limit_theorem(self,data,sample_size=30,total_samples=30):
        sample_data=[]
        for i in range(0,total_samples):
            sample_data.append(random.choices(data,k=sample_size))
        self.sample_means=[]
        for i in range(0,total_samples):
            self.sample_means.append(np.mean(sample_data[i]))
        self.smean=np.mean(self.sample_means)
        self.pmean=data.mean()
        self.svar=np.var(self.sample_means,ddof=1)
        self.pvar=data.var(ddof=0)/sample_size
        self.sstd=np.std(self.sample_means,ddof=1)
        self.pstd=data.std(ddof=0)/np.sqrt(sample_size)
        self.clt_result=pd.DataFrame({"Population":[self.pmean,self.pvar,self.pstd],"Sample":[self.smean,self.svar,self.sstd]},
                                   index=["Mean","Variance","Standard Deviation"])
        display(self.clt_result)
    def mean_plot(self):
        sns.distplot(self.sample_means)
    def display_results(self):
        display(self.clt_result)

def central_limit_theorem(data,sample_size=30,total_samples=30):
    '''
    This functions applys 
    
    '''
    
    pd.options.display.float_format = '{:,.2f}'.format
    obj_clt=clt()
    obj_clt.central_limit_theorem(data,sample_size,total_samples)
    return obj_clt
    

def confidence_interval(data,sample_size=40,confidence_level=95):
    sample_data=random.choices(data,k=sample_size)
    m=np.mean(sample_data)
    s=data.std()/np.sqrt(sample_size)
    confidence_level=confidence_level/100
    z=abs(st.norm.ppf((1-confidence_level)/2))
    lower_limit=m-(z*s)
    upper_limit=m+(z*s)
    return lower_limit,upper_limit
    
def find_outliers(dataset,method="iqr",cat_thresh=10,verbose=0,columns="all"):
    '''
    find_outliers function finds the outliers in the numerical data 
    and returns the percentage and no of ouliers present in each column
    
    Parameters
    ----------
    dataset : dataframe
        
    method: string, default iqr 
    
    
    cat_thresh:int, default 10
    
    
    
    
    verbose=int,default 0
            if verbose 1 prints the outliers
    
    columns:string,list or array 
    '''
    dataset=dataset.copy()
    dataset_num=pd.DataFrame()
    for col in dataset.columns:
        if((dataset[col].dtype=='int64' or dataset[col].dtype=='float64') and dataset[col].nunique()>cat_thresh):
             dataset_num[col]=dataset[col]
    if(method=="iqr"):
        if columns=="all":
            for col in dataset_num:
                q1=dataset_num[col].describe()["25%"]
                q3=dataset_num[col].describe()["75%"]
                iqr=q3-q1
                lb=q1-(1.5*iqr)
                ub=q3+(1.5*iqr)
                out=dataset_num[col][(dataset_num[col]<lb) | (dataset_num[col]>ub)]
                num_o=out.shape[0]
                p=(num_o/dataset_num.shape[0])*100
                if(num_o>0):
                    print("====================="+col+"===============")
                    print("Total",": ",num_o)
                    print("Percentage",": ",round(p,2),"%")
                    if(verbose):
                        display(out)
        else:
            for col in columns:
                if(col in dataset_num):
                    q1=dataset_num[col].describe()["25%"]
                    q3=dataset_num[col].describe()["75%"]
                    iqr=q3-q1
                    lb=q1-(1.5*iqr)
                    ub=q3+(1.5*iqr)
                    out=dataset_num[col][(dataset_num[col]<lb) | (dataset_num[col]>ub)]
                    num_o=out.shape[0]
                    p=(num_o/dataset_num.shape[0])*100
                    print("====================="+col+"===============")
                    print("Total",": ",num_o)
                    print("Percentage",": ",round(p,2),"%")
                    if(verbose):
                        display(out)
    if(method=="zscore"):
        if columns=="all":
            for col in dataset_num:
                num_o=(pd.DataFrame(abs(st.zscore(dataset[col])))>3).sum()[0]
                p=(num_o/dataset_num.shape[0])*100
                if(num_o>0):
                    print("====================="+col+"===============")
                    print("Total",": ",num_o)
                    print("Percentage",": ",round(p,2),"%")
        else:
            for col in columns:
                if(col in dataset_num):
                    num_o=(pd.DataFrame(abs(st.zscore(dataset[col])))>3).sum()[0]
                    p=(num_o/dataset_num.shape[0])*100
                    if(num_o>0):
                        print("====================="+col+"===============")
                        print("Total",": ",num_o)
                        print("Percentage",": ",round(p,2),"%")
        

    

     
def impute_missing(dataset,target=None,ignore_columns=None,numerical_imputation="mean",categorical_imputation="mode"):
    '''
    Due to various reasons data may have missing values. In general those missing values are handled by applying various strategies like imputation and removal. 
    Scalarpy  impute_missing function by default imputes the missing values in numerical column with “mean” and categorical column with “mode”.
    
    Parameters
    ----------
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string. The target variable could be continuous, binary or  multiclass.
    
    ignore_columns : list, default=None
    Columns to be ignored while imputing the missing values. To be passed in as a list.
    
    numerical_imputation : string, default=”mean”
    The strategy to be followed for imputing numerical columns.The other strategies available are “median”, “mode” .
    
    categorical_imputation : string, default=”mode”
    The strategy to be followed for imputing categorical columns.
    
    Returns
    ---------
    
    dataframe

    '''
    data=dataset.copy()
    obj_pp=preprocess(dataset,target=target,ignore_columns=ignore_columns)
    data=obj_pp.impute_missing(numerical_imputation,categorical_imputation)
    return data
def handle_outliers(dataset,target=None,ignore_columns=None,method="iqr",outlier_threshold=2,strategy="replace_lb_ub",columns="all"):
    '''
    An outlier is a data point that differs significantly from other observations, may be due to variability in the measurement or it may indicate experimental error. These outliers might affect your model performance.  Scalarpy handle_outliers function by default finds the outliers by using the IQR method and replaces the lower outliers with lower boundary and upper outliers with upper boundary.

    Parameters
    ----------
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string. The target variable could be continuous, binary or  multiclass.
    
    ignore_columns : list, default=None
    Columns to be ignored while handling the outliers. To be passed in as a list.
    
    method: string, default=”iqr”
    Strategy to be followed to find outliers, by default method “iqr”, the other method available zscore
    
    outlier_threshold : int (Percentage), default=2
    The threshold percentage to handle the outliers. For example if outlier_threshold is 2, the columns with less than 2 % of outliers are considered.
    
     strategy : string, default = replace_lb_ub(replaces the lower outliers with lower boundary and upper outliers with upper boundary)
    The strategy to be followed to handle the outliers, by default (replace_lb_ub) replaces the lower outliers with lower boundary and upper outliers with upper boundary, 
    Other strategies available 
    “remove” -- removes the outliers
    “replace_mean”- replaces outliers with mean
    “replace_median”-replaces outliers with median
    
    columns: string or list, default=”all”
    The list of columns to handle the outliers by default checks for all the columns
    
    Returns
    ---------
    
    dataframe

    '''
    data=dataset.copy()
    obj_pp=preprocess(dataset,target=target,ignore_columns=ignore_columns)
    data=obj_pp.handle_outliers(method,outlier_threshold,strategy,columns)
    return data    
def encode_data(dataset,target=None,ignore_columns=None,strategy="one_hot_encode",high_cardinality="frequency",drop_first=True,ordinal_map=None,categorical_features="auto",encode_map=None):
    '''
    As machine learning algorithms completely involve numerical computations, Categorical data cannot be used directly. So they must be transformed into numerical values before applying a model.
    In general many encoding strategies are used to convert categorical data into numerical format.
    Scalarpy encode_data function by default identifies the categorical data and encodes the categorical data into OneHotEncode format. 
    
    Parameters
    ----------
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string. The target variable could be continuous, binary or  multiclass.
    
    ignore_columns : list, default=None
    Columns to be ignored while encoding the categorical columns. To be passed in as a list.
    
    strategy: string, default =one_hot_encode
    Encoding Strategy to be followed to encode categorical columns, by default applies One Hot Encoding
    Other strategies available:
    
    label_encode -- applies Label Encoding to the categorical columns
    ordinal_encode -- Should be used when your categorical variable is ordinal type,. Applies Ordinal Encoding to the categorical columns,note that if ordinal encoding is used ordinal_map should not be empty
    frequency or count -- Frequency Encoding-replaces the categories with the frequency
    hybrid -- Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. For example you can apply one hot encoding to 1st and 2nd column and Label encoding to 4th column. Note that encode map should not be empty when this strategy is used
    
    high_cardinality: string, default=frequency
    High Cardinality are the type of variables with many categories. High_cardinality represents the strategy to be followed for encoding high_cardinality features
    
    drop_first: bool, default=True
    Works only when the strategy is “one_hot_encode”. If True, drops the first dummy variable to get k-1 dummies out of k categorical levels. 
    
    ordinal_map:dict, default=None
    Works only when the strategy is “ordinal_encode”.  Should be used when your categorical variable is ordinal type. ordinal_map should be provided in the form of a dictionary with key as column name and value as ordinal levels in the form a dictionary to encode that particular column. For example if you have a column called “performance” with categories 
    
    categorical_features : string or list, default=auto
    The list of categorical features to apply encoding, by default the list is automatically identified. If you want to manually encode particular columns, the column names should be passed in the form of list
    
    encode_map : dict, default =None
    Should be used only when the encoding strategy is “hybrid”. Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. encode_map  should be provided in the form of a dictionary with key as column name and value as encoding type.

    Returns
    ---------
    
    dataframe
    '''
    data=dataset.copy()
    obj_pp=preprocess(dataset,target=target,ignore_columns=ignore_columns)
    data=obj_pp.encode_data(strategy,high_cardinality,drop_first,ordinal_map,categorical_features,encode_map)
    return data 
def normalize(dataset,target=None,ignore_columns=None,method="min_max"):
    '''
    Parameters
    ----------
    
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string. Target will be ignored while normalizing. The target variable could be continuous, binary or  multiclass.
    
    ignore_columns : list, default=None
    Columns to be ignored while encoding the categorical columns. To be passed in as a list.
    
    method : string, default=”min_max”
    Strategy to be followed to normalize the data, by default Scalarpy normalize function applies Min Max Scaler 
    Other Strategies available :
    “standard” : applies Scikit Learn Standard scaler and converts the data into Standard Normal Distribution.
    “robust” : applies Scikit Learn Robust Scaler
    
    Returns
    ---------
    
    dataframe

    '''
    data=dataset.copy()
    obj_pp=preprocess(dataset,target=target,ignore_columns=ignore_columns)
    data=obj_pp.normalize(method)
    return data

def preprocess_data(dataset,target=None,ignore_columns=None,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    normalization_strategy="min_max"
                    ):
    '''
    Data preprocessing is the technique of preparing (cleaning and organizing) the raw data to make it suitable for building and training Machine Learning models. Before applying any machine learning algorithm, data preprocessing is a crucial step to clean, format and organize the raw data and make it ready-to-go for Machine Learning models. In general Data Scientists follow various strategies to preprocess the data and improve the model performance. 

    With Scalarpy’s preprocess_data function you can perform the entire data preprocessing steps at a single go by tweaking multiple parameters for various strategies based on your requirement.

    Parameters
    ----------
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string.The target variable could be continuous, binary or  multiclass.
    
    ignore_columns : list, default=None
    Columns to be ignored. To be passed in as a list.
    
    impute_missing : bool, default=True
    When set to True, missing values are handled based on the imputation strategy mentioned, set to False to ignore the missing values
    
    handle_outliers: bool, default=True
    When set to True, outliers are handled based on the outlier_method, set to False to ignore the outliers 
    
    encode_data : bool, default=True
    When set to True, categorical features are encoded based on the encoding_strategy, set to False to ignore this step
    
    normalize : bool, default=True
    When set to True, the data is normalized based on the normalization_strategy,set to False to ignore this step
    
    
    cat_thresh:int, default=10
    Threshold value which represents no of categories for identifying a categorical variable. For example if cat_thresh=10 then any variable more than 10 unique values is considered as a numerical variable 
    
    numerical_imputation : string, default=”mean”
    The strategy to be followed for imputing numerical variables.The other strategies available are “median”, “mode” .Works only when impute_missing =True.
    
    categorical_imputation : string, default=”mode”
    The strategy to be followed for imputing categorical variables.Works only when impute_missing =True.
    
    outlier_method: string, default=”iqr”
    Strategy to be followed to find outliers, by default method “iqr”, the other method available zscore
    
    outlier_threshold : int (Percentage), default=2
    The threshold percentage to handle the outliers. For example if outlier_threshold is 2, the columns with less than 2 % of outliers are considered.
    
     outlier_strategy: string, default = replace_lb_ub(replaces the lower outliers with lower boundary and upper outliers with upper boundary)
    The strategy to be followed to handle the outliers, by default (replace_lb_ub) replaces the lower outliers with lower boundary and upper outliers with upper boundary, 
    Other strategies available 
    “remove” -- removes the outliers
    “replace_mean”- replaces outliers with mean
    “replace_median”-replaces outliers with median
    
    outlier_columns: string or list, default=”all”
    The list of columns to handle the outliers by default checks for all the columns
    
    encoding_strategy: string, default =one_hot_encode
    Encoding Strategy to be followed to encode categorical columns, by default applies One Hot Encoding
    Other strategies available:
    
    label_encode -- applies Label Encoding to the categorical columns
    ordinal_encode -- Should be used when your categorical variable is ordinal type,. Applies Ordinal Encoding to the categorical columns,note that if ordinal encoding is used ordinal_map should not be empty
    frequency or count -- Frequency Encoding-replaces the categories with the frequency
    hybrid -- Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. For example you can apply one hot encoding to 1st and 2nd column and Label encoding to 4th column. Note that encode map should not be empty when this strategy is used
    
    high_cardinality_encoding: string, default=frequency
    High Cardinality are the type of variables with many categories. high_cardinality_encoding represents the strategy to be followed for encoding high_cardinality features
    
    encode_drop_first: bool, default=True
    Works only when the strategy is “one_hot_encode”. If True, drops the first dummy variable to get k-1 dummies out of k categorical levels. 
    
    ordinal_map:dict, default=None
    Works only when the strategy is “ordinal_encode”.  Should be used when your categorical variable is ordinal type. ordinal_map should be provided in the form of a dictionary with key as column name and value as ordinal levels in the form a dictionary to encode that particular column. For example if you have a column called “performance” with categories 
    
    encoding_categorical_features: string or list, default=auto
    The list of categorical features to apply encoding, by default the list is automatically identified. If you want to manually encode particular columns, the column names should be passed in the form of list
    
    encode_map : dict, default =None
    Should be used only when the encoding_strategy is “hybrid”. Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. encode_map  should be provided in the form of a dictionary with key as column name and value as encoding type.
     
    normalization_strategy: string, default=”min_max”
    Strategy to be followed to normalize the data, by default Min Max Scaler is applied.
    Other Strategies available :
    “standard” : applies Scikit Learn Standard scaler and converts the data into Standard Normal Distribution.
    “robust” : applies Scikit Learn Robust Scaler
    Returns
    ---------
    
    PreProcessed dataframe

    '''
    data=dataset.copy()
    obj_pp=preprocess(data,target,cat_thresh,ignore_columns)
    if impute_missing:
        obj_pp.impute_missing(numerical_imputation,categorical_imputation)
    if handle_outliers:
        obj_pp.handle_outliers(outlier_method,outlier_threshold,outlier_strategy,outlier_columns)
    if encode_data:
        obj_pp.encode_data(encoding_strategy,high_cardinality_encoding,encode_drop_first,ordinal_map,encoding_categorical_features,encode_map)
    if normalize:
        obj_pp.normalize(normalization_strategy)
    return obj_pp.data


def build_classifier(dataset,target=None,preprocess_data=True,classifiers="all",ignore_columns=None,train_size=0.8,random_state=42,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,sort="accuracy",
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    handle_imbalance=False,resampling_method="smote",
                    normalization_strategy="min_max",
                    hyperparameter_tunning="best",param_grid="auto",cv=10,n_iter=10, hyperparameter_scoring="accuracy",n_jobs=1,
                    verbose=1):
    '''
    Classification is a one of the supervised learning algorithms which is used to predict the categorical class label for example Diabetes Prediction, Churn Prediction etc.
    With Scalarpy’s build_classifier function you can build an entire Classification training pipeline starting from preprocessing to hyperparameter tuning  in one single step by tweaking various parameters according to your requirements.

    build_classifier function performs various steps starting from Data preprocessing, fitting multiple classification models, and hyperparameter tuning .
    This function returns the object of Scalarpy classifier which can be used further.
    
    Parameters
    ----------
    
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string.The target variable could be continuous, binary or  multiclass.
    
    
    preprocess_data : bool, default=True
    When set to True, the data is preprocessed by applying various techniques like impute missing values, handling outliers, encoding categorical columns, and normalizing the data
    
    classifiers : string or list, default=all
    List of classification models to be fitted. By default all the available classification models are fitted.
    
    
    ignore_columns : list, default=None
    Columns to be ignored. To be passed in as a list.
    
    train_size:float, default=0.8
    Size of the training set. By default, 80% of the data will be used for training. The remaining data will be used for a test
    
    random_state, int or RandomState instance, default=None
    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    
    
    impute_missing : bool, default=True
    When set to True, missing values are handled based on the imputation strategy mentioned, set to False to ignore the missing values
    
    handle_outliers: bool, default=True
    When set to True, outliers are handled based on the outlier_method, set to False to ignore the outliers 
    
    encode_data : bool, default=True
    When set to True, categorical features are encoded based on the encoding_strategy, set to False to ignore this step
    
    normalize : bool, default=True
    When set to True, the data is normalized based on the normalization_strategy,set to False to ignore this step
    
    
    cat_thresh:int, default=10
    Threshold value which represents no of categories for identifying a categorical variable. For example if cat_thresh=10 then any variable more than 10 unique values is considered as a numerical variable 
    
    numerical_imputation : string, default=”mean”
    The strategy to be followed for imputing numerical variables.The other strategies available are “median”, “mode” .Works only when impute_missing =True.
    
    categorical_imputation : string, default=”mode”
    The strategy to be followed for imputing categorical variables.Works only when impute_missing =True.
    
    outlier_method: string, default=”iqr”
    Strategy to be followed to find outliers, by default method “iqr”, the other method available zscore
    
    outlier_threshold : int (Percentage), default=2
    The threshold percentage to handle the outliers. For example if outlier_threshold is 2, the columns with less than 2 % of outliers are considered.
    
     outlier_strategy: string, default = replace_lb_ub(replaces the lower outliers with lower boundary and upper outliers with upper boundary)
    The strategy to be followed to handle the outliers, by default (replace_lb_ub) replaces the lower outliers with lower boundary and upper outliers with upper boundary, 
    Other strategies available 
    “remove” -- removes the outliers
    “replace_mean”- replaces outliers with mean
    “replace_median”-replaces outliers with median
    
    outlier_columns: string or list, default=”all”
    The list of columns to handle the outliers by default checks for all the columns
    
    encoding_strategy: string, default =one_hot_encode
    Encoding Strategy to be followed to encode categorical columns, by default applies One Hot Encoding
    Other strategies available:
    
    label_encode -- applies Label Encoding to the categorical columns
    ordinal_encode -- Should be used when your categorical variable is ordinal type,. Applies Ordinal Encoding to the categorical columns,note that if ordinal encoding is used ordinal_map should not be empty
    frequency or count -- Frequency Encoding-replaces the categories with the frequency
    hybrid -- Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. For example you can apply one hot encoding to 1st and 2nd column and Label encoding to 4th column. Note that encode map should not be empty when this strategy is used
    
    high_cardinality_encoding: string, default=frequency
    High Cardinality are the type of variables with many categories. high_cardinality_encoding represents the strategy to be followed for encoding high_cardinality features
    
    encode_drop_first: bool, default=True
    Works only when the strategy is “one_hot_encode”. If True, drops the first dummy variable to get k-1 dummies out of k categorical levels. 
    
    ordinal_map:dict, default=None
    Works only when the strategy is “ordinal_encode”.  Should be used when your categorical variable is ordinal type. ordinal_map should be provided in the form of a dictionary with key as column name and value as ordinal levels in the form a dictionary to encode that particular column. For example if you have a column called “performance” with categories 
    
    encoding_categorical_features: string or list, default=auto
    The list of categorical features to apply encoding, by default the list is automatically identified. If you want to manually encode particular columns, the column names should be passed in the form of list
    
    encode_map : dict, default =None
    Should be used only when the encoding_strategy is “hybrid”. Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. encode_map  should be provided in the form of a dictionary with key as column name and value as encoding type.
    
    handle_imbalance: bool,default=False
    Data imbalance usually reflects an unequal distribution of classes within a dataset,.When set to true the imbalanced data is handled by using the SMOTE by default. The strategy to handle the imbalanced data can be changed by using resampling_method parameter
    
    resampling_method : string,default=”smote”
    The strategy to handle the imbalanced data by default build_classifier uses Synthetic Minority Oversampling Technique to handle the imbalanced data
    Other Strategies available are:
    “over_sampler”: applies Random Oversampling Technique
    “under_sampler”: applies Random Undersampling Technique
     
    normalization_strategy: string, default=”min_max”
    Strategy to be followed to normalize the data, by default Min Max Scaler is applied.
    Other Strategies available :
    “standard” : applies Scikit Learn Standard scaler and converts the data into Standard Normal Distribution.
    “robust” : applies Scikit Learn Robust Scaler
    
    
    
    hyperparameter_tunning: string or list, default =”best”
    By default build_classifer applies hyperparameter_tunning using random search cv to the best model considering the scoring parameter. You can customise by passing the list of models to apply hyperparameter_tunning.
    
    
    param_grid: string, default =”auto”
    param_grid for hyperparameter tuning, If param_grid=”auto”
    per-configured hyperparameters for respective models are used.
    
    cv: int, cross-validation generator or an iterable, default=10
    Determines the cross-validation splitting strategy. Possible inputs for cv are:
    
    integer, to specify the number of folds in a (Stratified)KFold,
    
    CV splitter,
    An iterable yielding (train, test) splits as arrays of indices.
    
    n_iter: int, default=10
    Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    hyperparameter_scoring : string, callable, list/tuple or dict, default=”accuracy”
    A single str (see The scoring parameter: defining model evaluation rules) or a callable (see Defining your scoring strategy from metric functions) to evaluate the predictions on the test set.
    
    For evaluating multiple metrics, either give a list of (unique) strings or a dict with names as keys and callables as values.
    
    n_jobs: int, default=1
    Number of jobs to run in parallel. 1 means one unless in a joblib.parallel_backend context. -1 means using all processors.
    
    verbose:int default=1
    When verbose set to 1, prints the status of building process

    Returns
    ---------
    
    ScalarPy Classifier Object


    '''
    
    '''if(dataset==None or type(dataset)!=pd.core.frame.DataFrame):
        raise ValueError("Unsupported dataset type "+ str(type(dataset)))
    if(preprocess_data==None or type(preprocess_data)!='bool'):
        raise ValueError("Unsupported preprocess_data type "+ str(type(preprocess_data)))
    if(classifiers==None or type(classifiers)!='list' or type(classifiers)!='str'):
        raise ValueError("Unsupported classifiers type "+ str(type(classifiers)))'''
        
    cl=classifier(dataset,target,preprocess_data,classifiers,ignore_columns,train_size,random_state,impute_missing,handle_outliers,encode_data,normalize,sort,
                    numerical_imputation,categorical_imputation,cat_thresh,
                    outlier_method,outlier_threshold,outlier_strategy,outlier_columns,
                    encoding_strategy,high_cardinality_encoding,encode_drop_first,ordinal_map,encoding_categorical_features,encode_map,
                    handle_imbalance,resampling_method,
                    normalization_strategy,
                    hyperparameter_tunning,param_grid,cv,n_iter, hyperparameter_scoring,n_jobs,
                    verbose)
    cl.auto_classify(verbose)
    return cl


def build_regressor(dataset,target=None,preprocess_data=True,regressors="all",ignore_columns=None,train_size=0.8,random_state=42,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,sort="r2",
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    normalization_strategy="min_max",
                     hyperparameter_tunning="best",param_grid="auto",cv=10,n_iter=10, hyperparameter_scoring="r2",n_jobs=1,
                    verbose=1):
    '''
    Regression is one of the supervised machine learning types that is used for estimating the relationships between a dependent variable and one or more independent variables . The objective of regression is to predict continuous values such as predicting sales amount, predicting quantity, predicting temperature, etc.

    With Scalarpy’s build_regressor function you can build an entire regression training pipeline starting from preprocessing to hyperparameter tuning  in one single step by tweaking various parameters according to your requirements.
    
    Parameters
    ----------
    
    dataset : dataframe (Dataset).
    array-like, sparse matrix, shape (n_samples, n_features) where n_samples is the number of samples and n_features is the number of features.
    
    target : string, default=None
    Name of the target column to be passed in as a string.The target variable could be continuous, binary or  multiclass.
    
    
    preprocess_data : bool, default=True
    When set to True, the data is preprocessed by applying various techniques like impute missing values, handling outliers, encoding categorical columns, and normalizing the data
    
    regressors: string or list, default=all
    List of regression models to be fitted to the training data. By default all the available regression models are fitted.
    
    
    ignore_columns : list, default=None
    Columns to be ignored. To be passed in as a list.
    
    train_size:float, default=0.8
    Size of the training set. By default, 80% of the data will be used for training. The remaining data will be used for a test
    
    random_state, int or RandomState instance, default=None
    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    
    
    impute_missing : bool, default=True
    When set to True, missing values are handled based on the imputation strategy mentioned, set to False to ignore the missing values
    
    handle_outliers: bool, default=True
    When set to True, outliers are handled based on the outlier_method, set to False to ignore the outliers 
    
    encode_data : bool, default=True
    When set to True, categorical features are encoded based on the encoding_strategy, set to False to ignore this step
    
    normalize : bool, default=True
    When set to True, the data is normalized based on the normalization_strategy,set to False to ignore this step
    
    
    cat_thresh:int, default=10
    Threshold value which represents no of categories for identifying a categorical variable. For example if cat_thresh=10 then any variable more than 10 unique values is considered as a numerical variable 
    
    numerical_imputation : string, default=”mean”
    The strategy to be followed for imputing numerical variables.The other strategies available are “median”, “mode” .Works only when impute_missing =True.
    
    categorical_imputation : string, default=”mode”
    The strategy to be followed for imputing categorical variables.Works only when impute_missing =True.
    
    outlier_method: string, default=”iqr”
    Strategy to be followed to find outliers, by default method “iqr”, the other method available zscore
    
    outlier_threshold : int (Percentage), default=2
    The threshold percentage to handle the outliers. For example if outlier_threshold is 2, the columns with less than 2 % of outliers are considered.
    
     outlier_strategy: string, default = replace_lb_ub(replaces the lower outliers with lower boundary and upper outliers with upper boundary)
    The strategy to be followed to handle the outliers, by default (replace_lb_ub) replaces the lower outliers with lower boundary and upper outliers with upper boundary, 
    Other strategies available 
    “remove” -- removes the outliers
    “replace_mean”- replaces outliers with mean
    “replace_median”-replaces outliers with median
    
    outlier_columns: string or list, default=”all”
    The list of columns to handle the outliers by default checks for all the columns
    
    encoding_strategy: string, default =one_hot_encode
    Encoding Strategy to be followed to encode categorical columns, by default applies One Hot Encoding
    Other strategies available:
    
    label_encode -- applies Label Encoding to the categorical columns
    ordinal_encode -- Should be used when your categorical variable is ordinal type,. Applies Ordinal Encoding to the categorical columns,note that if ordinal encoding is used ordinal_map should not be empty
    frequency or count -- Frequency Encoding-replaces the categories with the frequency
    hybrid -- Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. For example you can apply one hot encoding to 1st and 2nd column and Label encoding to 4th column. Note that encode map should not be empty when this strategy is used
    
    high_cardinality_encoding: string, default=frequency
    High Cardinality are the type of variables with many categories. high_cardinality_encoding represents the strategy to be followed for encoding high_cardinality features
    
    encode_drop_first: bool, default=True
    Works only when the strategy is “one_hot_encode”. If True, drops the first dummy variable to get k-1 dummies out of k categorical levels. 
    
    ordinal_map:dict, default=None
    Works only when the strategy is “ordinal_encode”.  Should be used when your categorical variable is ordinal type. ordinal_map should be provided in the form of a dictionary with key as column name and value as ordinal levels in the form a dictionary to encode that particular column. For example if you have a column called “performance” with categories 
    
    encoding_categorical_features: string or list, default=auto
    The list of categorical features to apply encoding, by default the list is automatically identified. If you want to manually encode particular columns, the column names should be passed in the form of list
    
    encode_map : dict, default =None
    Should be used only when the encoding_strategy is “hybrid”. Hybrid Encoding is an advanced encoding strategy, where various encoding strategies can be applied for different columns. encode_map  should be provided in the form of a dictionary with key as column name and value as encoding type.
     
    normalization_strategy: string, default=”min_max”
    Strategy to be followed to normalize the data, by default Min Max Scaler is applied.
    Other Strategies available :
    “standard” : applies Scikit Learn Standard scaler and converts the data into Standard Normal Distribution.
    “robust” : applies Scikit Learn Robust Scaler
    
    
    hyperparameter_tunning: string or list, default =”best”
    By default build_classifer applies hyperparameter_tunning using random search cv to the best model considering the scoring parameter. You can customise by passing the list of models to apply hyperparameter_tunning.
    
    
    param_grid: string, default =”auto”
    param_grid for hyperparameter tuning, If param_grid=”auto”
    per-configured hyperparameters for respective models are used.
    
    cv: int, cross-validation generator or an iterable, default=10
    Determines the cross-validation splitting strategy. Possible inputs for cv are:
    
    integer, to specify the number of folds in a (Stratified)KFold,
    
    CV splitter,
    An iterable yielding (train, test) splits as arrays of indices.
    
    n_iter: int, default=10
    Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    hyperparameter_scoring : string, callable, list/tuple or dict, default=”accuracy”
    A single str (see The scoring parameter: defining model evaluation rules) or a callable (see Defining your scoring strategy from metric functions) to evaluate the predictions on the test set.
    
    For evaluating multiple metrics, either give a list of (unique) strings or a dict with names as keys and callables as values.
    
    n_jobs: int, default=1
    Number of jobs to run in parallel. 1 means one unless in a joblib.parallel_backend context. -1 means using all processors.
    
    verbose:int default=1
    When verbose set to 1, prints the status of building process
    
    Returns
    ---------
    
    ScalarPy Regressor Object

    '''
    re=regressor(dataset,target,preprocess_data,regressors,ignore_columns,train_size,random_state,impute_missing,handle_outliers,encode_data,normalize,sort,
                    numerical_imputation,categorical_imputation,cat_thresh,
                    outlier_method,outlier_threshold,outlier_strategy,outlier_columns,
                    encoding_strategy,high_cardinality_encoding,encode_drop_first,ordinal_map,encoding_categorical_features,encode_map,
                    normalization_strategy,
                    hyperparameter_tunning,param_grid,cv,n_iter, hyperparameter_scoring,n_jobs,
                    verbose)
    re.auto_regression(verbose)
    return re

def load(filename):
    '''
    load function loads a previously saved training pipeline from the current active directory into the current python environment. Load object must be a pickle file.

    Parameters
    ----------
    filename: string, default = none
    Name of pickle file to be passed as a string

    '''
    with open(filename,'rb') as f:
        model=pickle.load(f)
    return model
        


def load_model(filename):
    '''
    load_model function loads a previously saved specific model training pipeline from the current active directory into the current python environment. Load object must be a pickle file.

    Parameters
    ----------
    filename: string, default = none
    Name of pickle file to be passed as a string
    

    '''
    with open(filename,'rb') as f:
        model=pickle.load(f)
    return model
    
'''def get_classifiers(sub):
        sub.cl=cl'''
        


'''def choose_classifier(dataset,target,preprocess_data,classifiers,ignore_columns,train_size,random_state,impute_missing,handle_outliers,encode_data,normalize,sort,
                    numerical_imputation,categorical_imputation,cat_thresh,
                    outlier_method,outlier_threshold,outlier_strategy,outlier_columns,
                    encoding_strategy,high_cardinality_encoding,encode_drop_first,ordinal_map,encoding_categorical_features,encode_map,
                    handle_imbalance,resampling_method,
                    normalization_strategy,
                    hyperparameter_tunning,param_grid,cv,n_iter, hyperparameter_scoring,n_jobs,
                    verbose):
    if(classifiers=="select"):
        grid = GridspecLayout(4, 3)
        m1=['Logistic Regression', 'Ridge Classifier', 'K Neighbors Classifier']
        m2=['Linear SVM', 'Navie Bayes', 'Decision Tree Classifier']
        m3=['Random Forest Classifier', 'Ada Boost Classifier','Gradient Boosting Classifier']
        m4=['Extreme Boosting Classifier', 'Light Gradient Boosting Classifier']
        title=widgets.HTML(
        value="<b>Select Models</b>",
        )
        sub = widgets.Button(description="Submit",button_style='success')
        
        box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='80%')
        tb = widgets.HBox(children=[title],layout=box_layout)
        sb = widgets.HBox(children=[sub],layout=box_layout)
        
        for val,i in enumerate(m1):
            grid[0,val]=widgets.Checkbox(
            value=True,
            description=i,
            disabled=False,
            indent=False)
        for val,i in enumerate(m2):
            grid[1,val]=widgets.Checkbox(
            value=True,
            description=i,
            disabled=False,
            indent=False)
        for val,i in enumerate(m3):
            grid[2,val]=widgets.Checkbox(
            value=True,
            description=i,
            disabled=False,
            indent=False)
        for val,i in enumerate(m4):
            grid[3,val]=widgets.Checkbox(
            value=True,
            description=i,
            disabled=False,
            indent=False)
        display(tb)
        display(grid)
        display(sb)
        sub.on_click(get_classifiers,)
        return sub.cl'''

                    
                    
                
    
    
    
            
            

       

    
    



    
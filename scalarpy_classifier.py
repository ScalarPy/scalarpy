

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import math
from IPython.display import display,clear_output
import random
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as sk
import sklearn.model_selection as skm

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,recall_score,cohen_kappa_score,log_loss
from scalarpy.pre_process import preprocess
import ipywidgets as widgets
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.model_selection import LearningCurve

from yellowbrick.model_selection import CVScores
from yellowbrick.model_selection import FeatureImportances
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

import pickle



import warnings
warnings.filterwarnings('ignore') 


def highlight_max(s):
    '''
    highlight the maximum in a Series green.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_min(s):
    '''
    highlight the maximum in a Series green.
    '''
    is_min = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_min]

class classifier:
    '''
    build_classifier(dataset,target=None,preprocess_data=True,classifiers="all",ignore_columns=None,train_size=0.8,random_state=42,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    normalization_strategy="min_max",
                    hyperparameter_tunning="best",param_grid="auto",cv=10,n_iter=10, hyperparameter_scoring="accuracy",n_jobs=1,
                    verbose=1)
    '''
    def __init__(self,dataset,target=None,preprocess_data=True,classifiers="all",ignore_columns=None,train_size=0.8,random_state=42,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,sort="accuracy",
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    handle_imbalance=False,resampling_method="smote",
                    normalization_strategy="min_max",
                    hyperparameter_tunning="best",param_grid="auto",cv=10,n_iter=10, hyperparameter_scoring="accuracy",n_jobs=1,
                    verbose=1):
        self.target=target
        self.train_size=train_size
        self.random_state=random_state
        self.classifiers=classifiers
        self.pd=preprocess_data
        self.sort=sort
        self.handle_imbalance=handle_imbalance
        self.resampling_method=resampling_method
        
        self.hyperparameter_tunning=hyperparameter_tunning
        self.param_grid=param_grid
        self.cv=cv
        self.n_iter=n_iter
        self.n_jobs=n_jobs
        self.hyperparameter_scoring=hyperparameter_scoring
        
        
        
        if(preprocess_data):
            self.pp=preprocess(dataset,target,ignore_columns=ignore_columns)
            self.pp.preprocess_data(impute_missing,handle_outliers,encode_data,normalize,
                    numerical_imputation,categorical_imputation,cat_thresh,
                    outlier_method,outlier_threshold,outlier_strategy,outlier_columns,
                    encoding_strategy,high_cardinality_encoding,encode_drop_first,ordinal_map,encoding_categorical_features,encode_map,
                    normalization_strategy,verbose)
    def auto_classify(self,verbose=1):
        
            
        data=self.pp.data
        if(data[self.target].nunique()>2):
            self.c_type="multi_class"
        else:
            self.c_type="binary"
        X=data.drop(self.target,axis=1)
        y=data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test=skm.train_test_split(X,y,train_size=self.train_size,random_state=self.random_state)
        if(self.handle_imbalance):
            self.X_train,self.y_train=self.pp.handle_imbalance(self.X_train,self.y_train,self.resampling_method,verbose)
        #Logistic Regression
        self.models={}
        if(verbose):
            print("Part-2 Building the models...")
        classifiers=self.classifiers
        if(classifiers=="all" or ("lr" in classifiers)):
            self.lr=LogisticRegression()
            self.lr.fit(self.X_train,self.y_train)
            self.models["Logistic Regression"]=self.lr
        #Ridge Classififer
        if(classifiers=="all" or ("rc" in classifiers)):
            self.rc=RidgeClassifier()
            self.rc.fit(self.X_train,self.y_train)
            self.models["Ridge Classifier"]=self.rc
        #KNN
        if(classifiers=="all" or ("knn" in classifiers)):
            self.knn=KNeighborsClassifier()
            self.knn.fit(self.X_train,self.y_train)
            self.models["K Neighbors Classifier"]=self.knn
        #Decision Tree
        if(classifiers=="all" or ("dt" in classifiers)):
            self.dt=DecisionTreeClassifier()
            self.dt.fit(self.X_train,self.y_train)
            self.models["Decision Tree Classifier"]=self.dt
        #SVM
        if(classifiers=="all" or ("svm" in classifiers)):
            self.svm=SVC(kernel="linear")
            self.svm.fit(self.X_train,self.y_train)
            self.models["Linear SVM"]=self.svm
        #Navie Bayes
        if(classifiers=="all" or ("nb" in classifiers)):
            self.nb=GaussianNB()
            self.nb.fit(self.X_train,self.y_train)
            self.models["Navie Bayes"]=self.nb
        #Random Forest
        if(classifiers=="all" or ("rf" in classifiers)):
            self.rf=RandomForestClassifier()
            self.rf.fit(self.X_train,self.y_train)
            self.models["Random Forest Classifier"]=self.rf
        #ADA Boost
        if(classifiers=="all" or ("adb" in classifiers)):
            self.adb=AdaBoostClassifier()
            self.adb.fit(self.X_train,self.y_train)
            self.models["AdaBoost Classifier"]=self.adb
        #GBM
        if(classifiers=="all" or ("gbm" in classifiers)):
            self.gbm=GradientBoostingClassifier()
            self.gbm.fit(self.X_train,self.y_train)
            self.models["Gradient Boosting Classifier"]=self.gbm
        #XGBOOST
        if(classifiers=="all" or ("xgb" in classifiers)):
            self.xgb=XGBClassifier()
            self.xgb.fit(self.X_train,self.y_train)
            self.models["Extreme Boosting Classifier"]=self.xgb
        #lGBM
        if(classifiers=="all" or ("lgbm" in classifiers)):
            self.lgb=LGBMClassifier()
            self.lgb.fit(self.X_train,self.y_train)
            self.models["Light Gradient Boosting Classifier"]=self.lgb
        if(verbose):
            print(30*"=")
            print("Part-3 Evaluating Model Performance")
        #Evaluate Models
        score_grid=pd.DataFrame()
        for key,model in self.models.items():
            y_pred=model.predict(self.X_test)
            accuracy=accuracy_score(self.y_test,y_pred)
            auc=roc_auc_score(self.y_test,y_pred)
            precision=precision_score(self.y_test,y_pred)
            recall=recall_score(self.y_test,y_pred)
            f1=f1_score(self.y_test,y_pred)
            kappa=cohen_kappa_score(self.y_test,y_pred)
            logloss=log_loss(self.y_test,y_pred)
            score_dict={"Model":key,"Accuracy":accuracy,"AUC_ROC":auc,"Precision":precision,
                        "Recall":recall,"F1 Score":f1,"Kappa":kappa,"Log Loss":logloss}
            score_grid=score_grid.append( score_dict,ignore_index=True,sort=False)
        self.score_grid=score_grid.set_index('Model')
        if(self.hyperparameter_tunning=="best"):
            if(verbose):
                print(30*"=")
                print("Part-4 Tunning Hyperparameters")
            best=self.score_grid.sort_values(by="Accuracy",ascending=False).iloc[0,:].name
            tg=self.tune_model(m_model=best,param_grid=self.param_grid,cv=self.cv,n_iter=self.n_iter,scoring=self.hyperparameter_scoring,n_jobs=self.n_jobs)
            tune_grid=pd.DataFrame()
            for key,li in tg.items():
                model=li[0]
                y_pred=model.predict(self.X_test)
                accuracy=accuracy_score(self.y_test,y_pred)
                auc=roc_auc_score(self.y_test,y_pred)
                precision=precision_score(self.y_test,y_pred)
                recall=recall_score(self.y_test,y_pred)
                f1=f1_score(self.y_test,y_pred)
                kappa=cohen_kappa_score(self.y_test,y_pred)
                logloss=log_loss(self.y_test,y_pred)
                score_dict={"Model":key,"Accuracy":accuracy,"AUC_ROC":auc,"Precision":precision,
                            "Recall":recall,"F1 Score":f1,"Kappa":kappa,"Log Loss":logloss}
                tune_grid=tune_grid.append( score_dict,ignore_index=True,sort=False)
            self.tune_grid=tune_grid.set_index('Model')
        if(verbose):
            print(30*"=")
            print("Build Success")
        return self.score_grid
    def get_results(self):
        '''
        get_results function of the scalarpy classification module displays a table containing the scores of the model across the k-folds and also the hyperparameter tuning. Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, Kappa and Log Loss. 
        
        '''
        print("=============================Test Results===========================================")
        sg=self.score_grid
        tg=self.tune_grid
        if(sg.shape[0]>1):
            sg=sg.style.apply(highlight_max,subset=pd.IndexSlice[:, ["Accuracy","AUC_ROC","Precision","Recall","F1 Score","Kappa"]])
            sg=sg.apply(highlight_min,subset=pd.IndexSlice[:, ["Log Loss"]])
        if(self.hyperparameter_tunning and tg.shape[0]>1):
            tg=tg.style.apply(highlight_max,subset=pd.IndexSlice[:, ["Accuracy","AUC_ROC","Precision","Recall","F1 Score","Kappa"]])
            tg=tg.apply(highlight_min,subset=pd.IndexSlice[:, ["Log Loss"]])
        display(sg)
        print()
        print("======================Hyperparameter Tunning Results=================================")
        display(tg)
    def predict(self,data,classifiers="all"):
        '''
        Predict function of the scalarpy classification  predicts the output on the new input data provided. The function applies the entire preprocessing steps to the new data and returns the prediction by all the models by default.
        
        Parameters
        ----------
        data: dataframe or list
        The new input data for prediction
        
        classifiers: string or list, default=”all”
        The list of classifiers on which prediction should be applied on the new input data.By default predict function applies to the classifiers available in classification library

        '''
        if(self.pd):
            data=self.pp.preprocess_data_new(data)
        prediction_map={}
        for key,model in self.models.items():
            y_pred=model.predict(data)
            prediction_map[key]=y_pred
        return prediction_map
    def evaluate_model(self,model="lr"):
        '''
         evaluate_mode function of Scalarpy classification module displays the user interface for all of the available plots and models . returns a plot based on user selection.


        '''
        self.sm=widgets.Dropdown(options=[('Logistic Regression','lr'), ('Ridge Classifier','rc'), ('K Neighbors Classifier','knn'),
             ("Decision Tree Classifier",'dt'),("Linear SVM",'svm'),("Navie Bayes",'nb'),("Random Forest Classifier",'rf'),("Ada Boost Classifier",'adb'),
            ("Gradient Boosting Classifier",'gbm'),("Extreme Boosting Classifier",'xgb'),("Light Gradient Boosting Classifier",'lgb')],
                value=model,
                description='Select Model:',)
        self.plot_m=widgets.ToggleButtons(
                            options=[('ROC Curve', 'auc-roc'), 
                                     ('CV Scores', 'cv'), 
                                     ('Confusion Matrix', 'confusion_matrix'), 
                                     ('Precision Recall Curve', 'pc'),
                                     ('Class Prediction Error', 'cpr'),
                                     ('Classification Report', 'classification_report'),
                                     ('Learning Curve', 'learning'),
                                     ('Feature Importance', 'fi'),
                                     (' Discrimination Threshold', 'dt')
                                    ],

                            description='Select Evaluationt Technique:',

                            disabled=False,

                            button_style='', # 'success', 'info', 'warning', 'danger' or ''

                            icons=[''])
        display(self.sm)
        display(self.plot_m)
        self.plot(plot=self.plot_m.value,model=self.sm.value)
        self.plot_m.observe(self.call_plot,names=['value'])
        self.sm.observe(self.call_plot,names=['value'])
    def call_plot(self,change):
        #print(change)
        clear_output()
        display(self.sm)
        display(self.plot_m)
        self.plot(plot=self.plot_m.value,model=self.sm.value)


    def plot(self,plot,model=None):
        model= getattr(self,model)
        #print(model)
        if(plot == 'auc-roc'):
            visualizer = ROCAUC(model,support=True)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot == 'confusion_matrix'):
            visualizer = ConfusionMatrix(model,support=True)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="classification_report"):
            visualizer = ClassificationReport(model, support=True)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="learning"):
            visualizer = LearningCurve(model,cv=10, support=True)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="cv"):
            visualizer = CVScores(model,cv=10, support=True)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="pc"):
            visualizer = PrecisionRecallCurve(model)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="dt"):
            if(self.c_type=="binary"):
                visualizer = DiscriminationThreshold(model)
                visualizer.fit(self.X_train, self.y_train)       
                visualizer.score(self.X_test, self.y_test)        
                visualizer.show()  
            else:
                print("DiscriminationThreshold can be visuvalized only for Binary Classification")
        elif(plot=="cpr"):
                visualizer = ClassPredictionError(model)
                visualizer.fit(self.X_train, self.y_train)       
                visualizer.score(self.X_test, self.y_test)        
                visualizer.show()       
        elif(plot=="fi"):
                visualizer = FeatureImportances(model)
                visualizer.fit(self.X_train, self.y_train)       
                visualizer.score(self.X_test, self.y_test)        
                visualizer.show() 
    def tune_model(self,model=None,param_grid="auto",cv=10,n_iter=10, scoring="accuracy",n_jobs=1,m_model=None):
        model_list={'Logistic Regression': 'lr',
                     'Ridge Classifier': 'rc',
                     'K Neighbors Classifier': 'knn',
                     'Decision Tree Classifier': 'dt',
                     'Linear SVM': 'svm',
                     'Navie Bayes': 'nb',
                     'Random Forest Classifier': 'rf',
                     'Ada Boost Classifier': 'adb',
                     'Gradient Boosting Classifier': 'gbm',
                     'Extreme Boosting Classifier': 'xgb',
                     'Light Gradient Boosting Classifier': 'lgb'}
        if(m_model):
            model=model_list[m_model]
        tune_grid={}
        if("lr" in model):
            if (param_grid=="auto"):
                param_grid = {'C': np.arange(0, 10, 0.001),
                    "penalty": [ 'l1', 'l2'],
                    "class_weight": ["balanced", None]
                        }
            rsv=RandomizedSearchCV(estimator=LogisticRegression(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Logistic Regression"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("knn" in model):
            if (param_grid=="auto"):
                param_grid = {'n_neighbors': range(1,51),
                    'weights' : ['uniform', 'distance'],
                    'metric':["euclidean", "manhattan"]
                        }
            rsv=RandomizedSearchCV(estimator=KNeighborsClassifier(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["K Neighbors Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("dt" in model):
            if (param_grid=="auto"):
                param_grid = {"max_depth": np.random.randint(1, (len(self.X_train.columns)*.85),20),
                    "max_features": np.random.randint(1, len(self.X_train.columns),20),
                    "min_samples_split" : [2, 5, 10, 15, 100],
                    "min_samples_leaf":[2,3,4,5,6],
                    "criterion": ["gini", "entropy"]
                        }
            rsv=RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Decision Tree Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("nb" in model):
            if (param_grid=="auto"):
                 param_grid = {'var_smoothing': [0.000000001, 0.000000002, 0.000000005, 0.000000008, 0.000000009,
                                            0.0000001, 0.0000002, 0.0000003, 0.0000005, 0.0000007, 0.0000009, 
                                            0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.009,
                                            0.004, 0.005, 0.006, 0.007,0.008, 0.009, 0.01, 0.1, 1]
                        }
            rsv=RandomizedSearchCV(estimator=GaussianNB(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Navie Bayes"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("rf" in model):
            if (param_grid=="auto"):
                param_grid = {'n_estimators': list(np.linspace(start = 10, stop = 1200, num = 10,dtype="int")),
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }  
            rsv=RandomizedSearchCV(estimator=RandomForestClassifier(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Random Forest Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("ada" in model):
            if (param_grid=="auto"):
                param_grid = {'n_estimators':  np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'algorithm' : ["SAMME", "SAMME.R"]
                        }   
            rsv=RandomizedSearchCV(estimator=AdaBoostClassifier(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["AdaBoost Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("gbc" in model):
            if (param_grid=="auto"):
                 param_grid = {'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'subsample' : np.arange(0.1,1,0.05),
                        'min_samples_split' : [2,4,5,7,9,10],
                        'min_samples_leaf' : [1,2,3,4,5],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'max_features' : ['auto', 'sqrt', 'log2']
                        }     
            rsv=RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Gradient Boosting Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("lgb" in model):
            if (param_grid=="auto"):
                 param_grid = {'num_leaves': [10,20,30,40,50,60,70,80,90,100,150,200],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                        'n_estimators': [10, 30, 50, 70, 90, 100, 120, 150, 170, 200], 
                        'min_split_gain' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        }   
            rsv=RandomizedSearchCV(estimator=LGBMClassifier(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Light Gradient Boosting Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("xgb" in model):
            if (param_grid=="auto"):
                if self.c_type=="multi_class":
                    num_class = self.y_train.value_counts().count()
                    param_grid = {'learning_rate': np.arange(0,1,0.01),
                          'n_estimators': np.arange(10,500,20),
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                          'num_class' : [num_class, num_class]
                         }
                else:
                    param_grid = {'learning_rate': np.arange(0,1,0.01),
                          'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                          'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                          'colsample_bytree': [0.5, 0.7, 0.9, 1],
                          'min_child_weight': [1, 2, 3, 4],
                         }
            rsv=RandomizedSearchCV(estimator=XGBClassifier(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Extreme Boosting Classifier"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        self.tune_grid=tune_grid
        return self.tune_grid
    def save(self,filename):
        '''
        Save function of Scalarpy classification module saves the entire training pipeline  object into the current active directory as a pickle file for later use.
        
        Parameters
        ----------
        filename: string
        The name of the file to save the object

        '''
        with open(filename,'wb') as f:
            pickle.dump(self,f)
            print("Success!")
    def save_model(self,filename,model):
        model_dict={ 'lr':'Logistic Regression',
                      'rc':'Ridge Classifier',
                      'knn':'K Neighbors Classifier',
                      'dt':'Decision Tree Classifier',
                      'svm':'Linear SVM',
                      'nb':'Navie Bayes',
                      'rf':'Random Forest Classifier',
                      'adb':'Ada Boost Classifier',
                      'gbm':'Gradient Boosting Classifier',
                      'xgb':'Extreme Boosting Classifier',
                      'lgb':'Light Gradient Boosting Classifier'}
        rl=['X_test',
                 'X_train',
                 'adb',
                 'auto_classify',
                 'call_plot',
                 'cv',
                 'dt',
                 'evaluate_model',
                 'gbm',
                 'get_results',
                 'handle_imbalance',
                 'hyperparameter_scoring',
                 'hyperparameter_tunning',
                 'knn',
                 'lgb',
                 'lr',
                 'n_iter',
                 'n_jobs',
                 'nb',
                 'param_grid',
                 'plot',
                 'random_state',
                 'rc',
                 'resampling_method',
                 'rf',
                 'save',
                 'save_model',
                 'score_grid',
                 'sort',
                 'svm',
                 'train_size',
                 'tune_grid',
                 'tune_model',
                 'xgb',
                 'y_test',
                 'y_train']
        rl.remove(model)
        my_instance=self
        my_instance.models={model_dict[model]:getattr(my_instance,model)}
        for i in dir(my_instance):
            if(i in rl):
                if(type(getattr(my_instance,i))==type(getattr(my_instance,"auto_classify"))):
                    pass
                else:
                    delattr(my_instance,i)
        with open(filename,'wb') as f:
            pickle.dump(self,f)
            print("Success!")
            
            
        
        
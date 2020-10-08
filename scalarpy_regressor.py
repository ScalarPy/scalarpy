
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
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import TheilSenRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
import ipywidgets as widgets
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_squared_log_error
from scalarpy.pre_process import preprocess

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import AlphaSelection
from yellowbrick.model_selection import LearningCurve
from yellowbrick.regressor import CooksDistance
from yellowbrick.model_selection import CVScores
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.model_selection import ValidationCurve

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
class regressor:
    def __init__(self,dataset,target=None,preprocess_data=True,regressors="all",ignore_columns=None,train_size=0.8,random_state=42,impute_missing=True,handle_outliers=True,encode_data=True,normalize=True,sort="r2",
                    numerical_imputation="mean",categorical_imputation="mode",cat_thresh=10,
                    outlier_method="iqr",outlier_threshold=2,outlier_strategy="replace_lb_ub",outlier_columns="all",
                    encoding_strategy="one_hot_encode",high_cardinality_encoding="frequency",encode_drop_first=True,ordinal_map=None,encoding_categorical_features="auto",encode_map=None,
                    normalization_strategy="min_max",
                    hyperparameter_tunning="best",param_grid="auto",cv=10,n_iter=10, hyperparameter_scoring="r2",n_jobs=1,
                    verbose=1):
        self.target=target
        self.train_size=train_size
        self.random_state=random_state
        self.regressors=regressors
        self.pd=preprocess_data        
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
    def auto_regression(self,verbose=1):
        
            
        data=self.pp.data
        X=data.drop(self.target,axis=1)
        y=data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test=skm.train_test_split(X,y,train_size=self.train_size,random_state=self.random_state)
        
        if(verbose):
            print("Part-2 Building the models...")
        
        #Linear Regression
        self.models={}
        regressors=self.regressors
        if(regressors=="all" or ("lr" in regressors)):
            self.lr=LinearRegression()
            self.lr.fit(self.X_train,self.y_train)
            self.models["Linear Regression"]=self.lr
        #Ridge Regressor
        if(regressors=="all" or ("ridge" in regressors)):
            self.ridge=Ridge()
            self.ridge.fit(self.X_train,self.y_train)
            self.models["Ridge Regressor"]=self.ridge
        #Lasso Regressor
        if(regressors=="all" or ("lasso" in regressors)):
            self.lasso=Lasso()
            self.lasso.fit(self.X_train,self.y_train)
            self.models["Lasso Regressor"]=self.lasso
         #Lasso Regressor
        if(regressors=="all" or ("ts" in regressors)):
            self.ts=TheilSenRegressor()
            self.ts.fit(self.X_train,self.y_train)
            self.models["Theil Sen Regressor"]=self.ts
        #KNN
        if(regressors=="all" or ("knn" in regressors)):
            self.knn=KNeighborsRegressor()
            self.knn.fit(self.X_train,self.y_train)
            self.models["K Neighbors Regressor"]=self.knn
        #Decision Tree
        if(regressors=="all" or ("dt" in regressors)):
            self.dt=DecisionTreeRegressor()
            self.dt.fit(self.X_train,self.y_train)
            self.models["Decision Tree Regressor"]=self.dt
        #SVM
        if(regressors=="all" or ("svm" in regressors)):
            self.svm=SVR(kernel="linear")
            self.svm.fit(self.X_train,self.y_train)
            self.models["Linear SVR"]=self.svm
        #Navie Bayes
        #Random Forest
        if(regressors=="all" or ("rf" in regressors)):
            self.rf=RandomForestRegressor()
            self.rf.fit(self.X_train,self.y_train)
            self.models["Random Forest Regressor"]=self.rf
        #ADA Boost
        if(regressors=="all" or ("adb" in regressors)):
            self.adb=AdaBoostRegressor()
            self.adb.fit(self.X_train,self.y_train)
            self.models["Ada Boost Regressor"]=self.adb
        #GBM
        if(regressors=="all" or ("gbm" in regressors)):
            self.gbm=GradientBoostingRegressor()
            self.gbm.fit(self.X_train,self.y_train)
            self.models["Gradient Boosting Regressor"]=self.gbm
        #XGBOOST
        if(regressors=="all" or ("xgb" in regressors)):
            self.xgb=XGBRegressor()
            self.xgb.fit(self.X_train,self.y_train)
            self.models["Extreme Boosting Regressor"]=self.xgb
        #lGBM
        if(regressors=="all" or ("lgb" in regressors)):
            self.lgb=LGBMRegressor()
            self.lgb.fit(self.X_train,self.y_train)
            self.models["Light Gradient Boosting Regressor"]=self.lgb
        if(verbose):
            print(30*"=")
            print("Part-3 Evaluating Model Performance")
        #Evaluate Models
        score_grid=pd.DataFrame()
        for key,model in self.models.items():
            y_pred=model.predict(self.X_test)
            mse=mean_squared_error(self.y_test,y_pred)
            mae=mean_absolute_error(self.y_test,y_pred)
            mle=mean_squared_log_error(self.y_test,y_pred)
            r2=r2_score(self.y_test,y_pred)
            rmse=np.sqrt(mse)
            rmsle=np.sqrt(mle)
            score_dict={"Model":key,"MSE":mse,"MAE":mae,"MLE":mle,
                        "R2_Score":r2,"RMSE":rmse,"RMSLE":rmsle}
            score_grid=score_grid.append( score_dict,ignore_index=True,sort=False)
        self.score_grid=score_grid.set_index('Model')
        
        if(self.hyperparameter_tunning=="best"):
            if(verbose):
                print(30*"=")
                print("Part-4 Tunning Hyperparameters")
            best=self.score_grid.sort_values(by="R2_Score",ascending=False).iloc[0,:].name
            tg=self.tune_model(m_model=best,param_grid=self.param_grid,cv=self.cv,n_iter=self.n_iter,scoring=self.hyperparameter_scoring,n_jobs=self.n_jobs)
            tune_grid=pd.DataFrame()
            for key,li in tg.items():
                model=li[0]
                y_pred=model.predict(self.X_test)
                mse=mean_squared_error(self.y_test,y_pred)
                mae=mean_absolute_error(self.y_test,y_pred)
                mle=mean_squared_log_error(self.y_test,y_pred)
                r2=r2_score(self.y_test,y_pred)
                rmse=np.sqrt(mse)
                rmsle=np.sqrt(mle)
                score_dict={"Model":key,"MSE":mse,"MAE":mae,"MLE":mle,
                            "R2_Score":r2,"RMSE":rmse,"RMSLE":rmsle}
                tune_grid=tune_grid.append( score_dict,ignore_index=True,sort=False)
            self.tune_grid=tune_grid.set_index('Model')     

        
        if(verbose):
            print(30*"=")
            print("Build Success")
        return self.score_grid
    def predict(self,data,regressors="all"):
        '''
        Predict function of the scalarpy regression model  predicts the output on the new input data provided. The function applies the entire preprocessing steps to the new data and returns the prediction by all the models by default.

        Parameters
        ----------
        data: dataframe or list
        The new input data for prediction
        
        regressors: string or list, default=”all”
        The list of regressors on which prediction should be applied on the new input data.By default predict function applies to the regressors available in regression library 


        '''
        if(self.pd):
            data=self.pp.preprocess_data_new(data)
        prediction_map={}
        for key,model in self.models.items():
            y_pred=model.predict(data)
            prediction_map[key]=y_pred
        return prediction_map
    def get_results(self):
        '''
        get_results function of the scalarpy regression module displays A table containing the scores of the model across the k-folds. Scoring metrics used are MAE, MSE, RMSE, R2, RMSLE and MLE
        
        
        '''
        sg=self.score_grid
        tg=self.tune_grid
        if(sg.shape[0]>1):
            sg=sg.style.apply(highlight_min,subset=pd.IndexSlice[:, ["MSE","MAE","MLE","RMSE","RMSLE"]])
            #display(type(self.score_grid))
            sg=sg.apply(highlight_max,subset=pd.IndexSlice[:, ["R2_Score"]])
        if(self.hyperparameter_tunning and tg.shape[0]>1):
            tg=tg.style.apply(highlight_min,subset=pd.IndexSlice[:, ["MSE","MAE","MLE","RMSE","RMSLE"]])
            #display(type(self.score_grid))
            tg=tg.apply(highlight_max,subset=pd.IndexSlice[:, ["R2_Score"]])
        print("=============================Test Results===========================================")
        display(sg)
        print()
        print("======================Hyperparameter Tunning Results=================================")
        display(tg)
    def evaluate_model(self,model="lr"):
        '''
        evaluate_mode function of Scalarpy regression module displays the user interface for all of the available plots and models . returns a plot based on user selection.
        '''
        self.sm=widgets.Dropdown(options=[('Linear Regression','lr'), ('Ridge Regressor','ridge'), ('Lasso Regressor','lasso'),
             ("Theil Sen Regressor",'ts'),("K Neighbors Regressor",'knn'),
             ("Decision Tree Regressor",'dt'),("Linear SVR",'svm'),("Random Forest Regressor",'rf'),
            ("Ada Boost Regressor",'adb'),("Gradient Boosting Regressor",'gbm'),("Extreme Boosting Regressor",'xgb'),
            ("Light Gradient Boosting Regressor",'lgb')],
                value=model,
                description='Select Model:',)
        self.plot_m=widgets.ToggleButtons(
                            options=[('Residuals Plot', 'residuals'), 
                                     ('Prediction Error Plot', 'pep'), 
                                     ('Cooks Distance Plot', 'cdp'),
                                     ('Alpha Selection Plot', 'asp'),
                                     ('Learning Curve', 'learning'),
                                     ('Validation Curve', 'vc'),
                                     ('Feature Importance', 'fi')],

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
        if(plot == 'residuals'):
            visualizer = ResidualsPlot(model)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot == 'pep'):
            visualizer = PredictionError(model)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot == 'learning'):
            visualizer = LearningCurve(model)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot == 'vc'):
            visualizer = ValidationCurve(model)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="fi"):
            visualizer = FeatureImportances(model)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
        elif(plot=="cv"):
            visualizer = CVScores(model,cv=10, support=True)
            visualizer.fit(self.X_train, self.y_train)       
            visualizer.score(self.X_test, self.y_test)        
            visualizer.show()
    def tune_model(self,model=None,param_grid="auto",cv=10,n_iter=10, scoring="r2",n_jobs=1,m_model=None):
        model_list={'Linear Regression': 'lr',
                     'Ridge Regressor': 'ridge',
                     'Lasso Regressor': 'lasso',
                     'Theil Sen Regressor': 'ts',
                     'K Neighbors Regressor': 'knn',
                     'Decision Tree Regressor': 'dt',
                     'Linear SVR': 'svm',
                     'Random Forest Regressor': 'rf',
                     'Ada Boost Regressor': 'adb',
                     'Gradient Boosting Regressor': 'gbm',
                     'Extreme Boosting Regressor': 'xgb',
                     'Light Gradient Boosting Regressor': 'lgb'}
        if(m_model):
            model=model_list[m_model]
        tune_grid={}
        if("lr" in model):
            if (param_grid=="auto"):
                param_grid = {'fit_intercept': [True, False],
                        'normalize' : [True, False]
                        }
            rsv=RandomizedSearchCV(estimator=LinearRegression(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Linear Regression"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("ridge" in model):
            if (param_grid=="auto"):
                param_grid = {"alpha": np.arange(0,1,0.001),
                        "fit_intercept": [True, False],
                        "normalize": [True, False],
                        }
            rsv=RandomizedSearchCV(estimator=Ridge(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Ridge Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("ts" in model):
            if (param_grid=="auto"):
                param_grid = {'fit_intercept': [True, False],
                        'max_subpopulation': [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
                        } 
            rsv=RandomizedSearchCV(estimator=TheilSenRegressor(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Theil Sen Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("lasso" in model):
            if (param_grid=="auto"):
                param_grid = {'alpha': np.arange(0,1,0.001),
                        'fit_intercept': [True, False],
                        'normalize' : [True, False],
                        }
            rsv=RandomizedSearchCV(estimator=Lasso(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Lasso Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("knn" in model):
            if (param_grid=="auto"):
                param_grid = {'n_neighbors': range(1,51),
                        'weights' :  ['uniform', 'distance'],
                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [10,20,30,40,50,60,70,80,90]
                        } 
            rsv=RandomizedSearchCV(estimator=KNeighborsRegressor(), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["K Neighbors Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("dt" in model):
            if (param_grid=="auto"):
                param_grid = {"max_depth": np.random.randint(1, (len(self.X_train.columns)*.85),20),
                    "max_features": np.random.randint(1, len(self.X_train.columns),20),
                    "min_samples_split" : [2, 5, 10, 15, 100],
                    "min_samples_leaf":[2,3,4,5,6],
                    "criterion": ["mse", "mae", "friedman_mse"]
                        }
            rsv=RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Decision Tree Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("rf" in model):
            if (param_grid=="auto"):
                param_grid = {'n_estimators': list(np.linspace(start = 10, stop = 1200, num = 10,dtype="int")),
                         "criterion": ["mse", "mae", "friedman_mse"],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'min_samples_split': [2, 5, 7, 9, 10],
                        'min_samples_leaf' : [1, 2, 4],
                        'max_features' : ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False]
                        }  
            rsv=RandomizedSearchCV(estimator=RandomForestRegressor(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Random Forest Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("adb" in model):
            if (param_grid=="auto"):
                param_grid = {'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0.1,1,0.01),
                        'loss' : ["linear", "square", "exponential"]
                        }  
            rsv=RandomizedSearchCV(estimator=AdaBoostRegressor(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["AdaBoost Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("gbm" in model):
            if (param_grid=="auto"):
                 param_grid = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                        'n_estimators': np.arange(10,200,5),
                        'learning_rate': np.arange(0,1,0.01),
                        'subsample' : [0.1,0.3,0.5,0.7,0.9,1],
                        'criterion' : ['friedman_mse', 'mse', 'mae'],
                        'min_samples_split' : [2,4,5,7,9,10],
                        'min_samples_leaf' : [1,2,3,4,5,7],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'max_features' : ['auto', 'sqrt', 'log2']
                        }     
            rsv=RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Gradient Boosting Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
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
            rsv=RandomizedSearchCV(estimator=LGBMRegressor(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Light Gradient Boosting Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        if("xgb" in model):
            if (param_grid=="auto"):
                param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                        'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                        'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
                        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
                        'colsample_bytree': [0.5, 0.7, 0.9, 1],
                        'min_child_weight': [1, 2, 3, 4]
                        }
            rsv=RandomizedSearchCV(estimator=XGBRegressor(random_state=self.random_state), 
                                        param_distributions=param_grid, scoring=scoring, n_iter=n_iter, cv=cv, 
                                        random_state=self.random_state, n_jobs=n_jobs)
            rsv.fit(self.X_train,self.y_train)
            tune_grid["Extreme Boosting Regressor"]=[rsv.best_estimator_,rsv.best_params_,rsv.best_score_]
        self.tune_grid=tune_grid
        return self.tune_grid       
    def save(self,filename):
        '''
        Save function of Scalarpy regression module saves the entire training pipeline  object into the current active directory as a pickle file for later use.

        Parameters
        ----------
        filename: string
        The name of the file to save the object


        '''
        with open(filename,'wb') as f:
            pickle.dump(self,f)
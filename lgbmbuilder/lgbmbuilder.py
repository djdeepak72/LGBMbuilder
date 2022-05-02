import math
import pickle
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, explained_variance_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_curve)
from sklearn.model_selection import train_test_split

__all__ = ['create_min_var', 'create_max_var', 'read_from_s3', 'data_preparation', 'lgbm_data', 'create_data_structure',
           'train_lgbm', 'actual_vs_predicted', 'plotting', 'performance', 'model_comparison', 
           'hyperparameter_tuning', 'add_deltamin', 'add_predictor','pr_curve_plot','roc_curve_plot','caliberation_plot'
           ]

def data_preparation(file_path='', output_filepath='', save_data=False,
              df=pd.DataFrame(), to_categorical=False,
              trainTestSplit=False, pc_testingData=0.3, pc_tuningData=0.3, random_seed=45, show_proportions=False):

    if output_filepath=='':
        output_filepath = file_path

    if df.empty:
        print('Reading data from path: ',file_path)
        if file_path=='':
            raise ValueError("Error - file_path not defined, please specify one")
        else:
            path, file_type = os.path.splitext(file_path)
        if file_type=='.csv':
            data=pd.read_csv(file_path)
        elif file_type=='.xlsx':
            data=pd.read_excel(file_path)
        elif file_type=='.parquet':
            data=pd.read_parquet(file_path, engine='pyarrow')
        else:
            raise ValueError("Error - file type must be one of ['.csv','.xlsx' or'.parquet']. We got: "+file_type+" instead")
    else:
        print('Using the data passed in `df` argument')
        data=df.copy()

    if to_categorical:
        print('Variables of `object` datatype found. Converting them to categorical')
        for col in data.columns.values:
            if data[col].dtypes == 'O':
                data.loc[:,col] = data.loc[:,col].astype('category')
                    
    if trainTestSplit:
        print('Creating training, test & validation sets')
        modellingdata = data.copy()
        train, test = train_test_split(modellingdata, test_size=pc_testingData, random_state = random_seed)
        test, validation = train_test_split(test, test_size=pc_tuningData, random_state = random_seed)

        if show_proportions:
            print('Modelling data proportions:')
            print('Training data:',"{:.1%}".format(train.shape[0]/data.shape[0]))
            print('Testing data:',"{:.1%}".format(test.shape[0]/data.shape[0]))
            print('Validation data:',"{:.1%}".format(validation.shape[0]/data.shape[0]))
    
    if save_data:
        print('Saving training, testing & validation data to : '+output_filepath)
        if file_type=='.parquet':
            print("Saving as .parquet files")
            train.to_parquet(output_filepath +'train.parquet')
            test.to_parquet(output_filepath +'test.parquet')
            validation.to_parquet(output_filepath +'validation.parquet')
        else:
            print("Saving as .csv files")
            train.to_csv(output_filepath +'train.csv')
            test.to_csv(output_filepath +'test.csv')
            validation.to_csv(output_filepath +'validation.csv')

    if trainTestSplit:
        return {'all':data,  'train':train, 'test':test, 'valid':validation}
    else:
        return {'all':data}
    

def create_data_structure(predictor_cols, response_col='', use_all_data=False, data={}):
    train=data['train'].copy()
    test=data['test'].copy()
    valid=data['valid'].copy()
    
    x_train=train[predictor_cols]
    x_test=test[predictor_cols]
    x_valid=valid[predictor_cols]
    
    categorical_data=list(x_train.select_dtypes(include='category').columns.values)
    
    result = {}
    
    if len(response_col) != 0:
        y_train=train[response_col]
        y_test=test[response_col]
        y_valid=valid[response_col]
        if use_all_data:
            y_all=pd.concat([y_train, y_test, y_valid])
            result.update({'y': y_all})
        else:
            result.update({'y_train':y_train, 'y_test':y_test, 'y_valid':y_valid})
    
    if use_all_data:
        x_all=pd.concat([x_train, x_test, x_valid])
        result.update({'X':x_all, 'categorical_data':categorical_data})
        return result
    
    else:
        result.update({'x_train':x_train, 'x_test':x_test, 'x_valid':x_valid, 
                            'categorical_data':categorical_data})
        return result

def lgbm_data(data, use_all_data=False):
    
    if use_all_data:
        lgb_all = lgb.Dataset(data['X'], label=data['y'],
                        feature_name=list(data['X'].columns.values),
                        categorical_feature=data['categorical_data'], 
                        free_raw_data=False
                              )
        return {'lgb_all':lgb_all, 'input_data':data}

    else:
        lgb_train = lgb.Dataset(data['x_train'], label=data['y_train'],
                        feature_name=list(data['x_train'].columns.values),
                        categorical_feature=data['categorical_data'], 
                        free_raw_data=False
                              )
        lgb_test = lgb.Dataset(data['x_test'], label=data['y_test'],
                        feature_name=list(data['x_test'].columns.values),
                        categorical_feature=data['categorical_data'], 
                        free_raw_data=False
                              )
        lgb_valid = lgb.Dataset(data['x_valid'], label=data['y_valid'],
                        feature_name=list(data['x_valid'].columns.values),
                        categorical_feature=data['categorical_data'], 
                        free_raw_data=False
                              )

        return {'lgb_train':lgb_train, 'lgb_test':lgb_test, 'lgb_valid':lgb_valid, 
                'input_data':data}
    
    
def train_lgbm(data, params, do_param_tuning=False, n_trials=None, max_runtime=None,
               num_boost_round=10000, verbose_eval=100):
    
    ##Peform hyperparameter tuning if specified & update the `params` argument when done.
    if do_param_tuning:
        tuned_params = hyperparameter_tuning(data, metric=params['metric'], n_trials=n_trials, max_runtime_hrs=max_runtime)
        params.update(tuned_params)
    
    evals_result = {} 
    if 'lgb_all' in data: #training using all the data
        model = lgb.train(params, data['lgb_all'], num_boost_round = num_boost_round)
    else: 
        model = lgb.train(params, data['lgb_train'], 
                        valid_sets = [data['lgb_train'],data['lgb_test']],
                        valid_names = ['train', 'val'], 
                        num_boost_round = num_boost_round, evals_result=evals_result,
                        verbose_eval=verbose_eval)
        
    return {'model':model, 'params':params, 'evals_result':evals_result, 'lgbm_Dataset':data}


def hyperparameter_tuning(data_dict, metric='rmse',early_stopping_rounds=10,num_boost_round=1000, n_trials=None, n_jobs=1, max_runtime_hrs=None):
    
    if metric not in ['rmse','binary_logloss']:
        raise ValueError("Error - only ['rmse','binary_logloss'] metrics are supported currently")
    
    maxtime=max_runtime_hrs
    if max_runtime_hrs!=None:
        maxtime=max_runtime_hrs*3600
        
    elif metric=='rmse':
        def optimise_fn(trial):

            param = {
                'boosting_type': 'gbdt', 
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'bagging_freq': 1,
                'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0, 5),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 2, 600),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 800),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1, 20),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1, 20),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            }

            gbm = lgb.train(param, data_dict['lgb_train'], valid_sets=[data_dict['lgb_test']], valid_names = ['test'],
                                early_stopping_rounds=early_stopping_rounds, num_boost_round=num_boost_round, verbose_eval=False)
            bst_score = gbm.best_score['valid']['rmse']
            return bst_score

        optuna.logging.set_verbosity(0)
        tune = optuna.create_study(direction='minimize')
        tune.optimize(optimise_fn,n_trials=n_trials,n_jobs=n_jobs,timeout=maxtime)
        best=tune.best_trial.params
        print('tuned_params:',best)

        return best

    elif metric=='binary_logloss':
        def optimise_fn(trial):

            param = {
                'boosting_type': 'gbdt', 
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbose': -1,
                'bagging_freq': 1,
                'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0, 5),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 2, 600),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 800),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1, 20),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1, 20),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            }

            gbm = lgb.train(param, data_dict['lgb_train'], valid_sets=[data_dict['lgb_test']], valid_names = ['valid'],
                            early_stopping_rounds=early_stopping_rounds, num_boost_round=num_boost_round, verbose_eval=False)
            bst_score = gbm.best_score['valid']['binary_logloss']
            return bst_score

        optuna.logging.set_verbosity(0)
        tune = optuna.create_study(direction='minimize')
        tune.optimize(optimise_fn,n_trials=n_trials,timeout=maxtime)
        best=tune.best_trial.params
        print('tuned_params:',best)

        return best

def actual_vs_predicted(df, prediction, variable, categorical=False, variable_max=-1, n_bins=10, step=2, qcut=False, plot_title='', 
                 rd_x=True, plot_distribution=False):
    weight=np.ones(df.shape[0])
    df1 = pd.DataFrame({variable:df[variable], "actual":df.actual, "weight":weight, "preds":prediction})
    
    if variable_max != -1 and categorical==False:
        df1 = df1[df1[variable]<variable_max]
        
    #bin across premium, create a list of each bin's midpoint (to be used in plot)
    if categorical==False:
        
        if qcut:
            df1['bin'], bins = pd.qcut(df1[variable], q = n_bins, retbins=True)
        else:
            df1['bin'], bins = pd.cut(df1[variable], bins = n_bins, retbins=True)
        
        if rd_x:
            mid = ["{:,}".format(int(round((a + b) /2,-1))) for a,b in zip(bins[:-1], bins[1:])]
        else:
            mid = ["{:,}".format(round((a + b) /2,1)) for a,b in zip(bins[:-1], bins[1:])]
    
    else:
        df1['bin']=df1[variable]
        mid = list(df1[variable])
        
    result = df1.groupby(df1.bin).agg('sum').reset_index()
    result['Pred'] = result['preds']/result['weight']
    result['Act'] = result['actual']/result['weight']
    result['Error'] = result['Pred']/result['Act'] - 1
    result['Dist'] = result['weight']/sum(result['weight'])
    result['Zeros'] = 0
    
    a = int(math.ceil(n_bins/step))
    labels = list()
    for i in range(0,a):
        l = i*step
        labels.append(l)
    tick_labels = list( mid[j] for j in labels)

    #plot
    if categorical==False:
        result['bin'] = result.bin.cat.codes
    if plot_distribution:
        y_tmp = result['Distribution']
    else:
        y_tmp = result['Error']
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.plot(result['bin'],result['Act'],color='y',label='Actual')
    ax.plot(result['bin'],result['Pred'],color='k',label='Predictions')
    if plot_distribution:
        ax1.plot(result['bin'],y_tmp,color='r',linestyle = '--',label='Distribution')
    else:
        ax1.plot(result['bin'],y_tmp,color='r',linestyle = '--',label='Percent error')
    ax1.plot(result['bin'],result['Zeros'],color='grey',linestyle = ':')
    ax.xaxis.set_ticks(np.arange(0, len(result['Zeros']), step))
    ax.set_xticklabels(tick_labels)
    ax.set_title(plot_title)
    ax.set_xlabel(variable)
    ax.set_ylabel('Predicted vs Actual Values')
    if plot_distribution:
        ax1.set_ylabel('Distribution')
    else:
        ax1.set_ylabel('Percent error')
    ax.legend(loc="upper right")
    plt.xticks(rotation=45)
    plt.show()
    
    return result

def plotting(df, variables, qcut, plot_title=''):
    for feature in variables:
        if qcut:
            if feature in list(df.select_dtypes(include='category').columns.values):
                bins = len(df[feature].unique())
                _ = actual_vs_predicted(df=df, prediction=df.preds, variable=feature, 
                     step = 1, n_bins = bins, qcut=True, categorical=True, rd_x=True, plot_distribution=False,
                     plot_title=plot_title)
            else:
                _ = actual_vs_predicted(df=df, prediction=df.preds, variable=feature, 
                     step = 2, n_bins = 10, qcut=True, categorical=False, rd_x=True, plot_distribution=False,
                     plot_title=plot_title)
        else:
            if feature in list(df.select_dtypes(include='category').columns.values):
                bins = len(df[feature].unique())
                _ = actual_vs_predicted(df=df, prediction=df.preds, variable=feature, 
                     step = 1, n_bins = bins, qcut=False, categorical=True, rd_x=True, plot_distribution=False,
                     plot_title=plot_title)
            else:
                _ = actual_vs_predicted(df=df, prediction=df.preds, variable=feature, 
                     step = 2, n_bins = 10, qcut=False, categorical=False, rd_x=True, plot_distribution=False,
                     plot_title=plot_title)

def performance(model_dict, predict_on=["test","valid"], 
                training_progress=False, feature_importance=False, percent_data_shap=0.0, 
                pr_curve_test=False, pr_curve_valid=False,
                roc_curve_test=False, roc_curve_valid=False,
                caliberation_plot_test=False, caliberation_plot_valid=False,
                actual_v_pred_test=False, actual_v_pred_valid=False, variableList=[], qcut=False
                ):
    
    
    if training_progress:
        ax = lgb.plot_metric(model_dict['evals_result'])
        plt.show()
    
    if feature_importance:
        ax = lgb.plot_importance(model_dict['model'], importance_type='gain', max_num_features=20)
        plt.show()
    
    if percent_data_shap<=0 or percent_data_shap>1:
        print(" `percent_data_shap` should be within 0 and 1. Skipping shap plot")
    else:
        sample=model_dict['lgbm_Dataset']['input_data']['x_train'].sample(n=5000, random_state=45)
        plotdf=sample.copy()
        if model_dict['model'].params['objective'] == 'binary':
            model_dict['model'].params['objective'] = 'binary_logloss'            
        explainer = shap.TreeExplainer(model_dict['model'])
        shapdf = explainer.shap_values(sample)
        shap.summary_plot(shapdf, plotdf)
    
    if "train" in predict_on:
        model_dict['prediction_train'] = model_dict['model'].predict(model_dict['lgbm_Dataset']['input_data']['x_train'])
    
    if "test" in predict_on:
        model_dict['prediction_test'] = model_dict['model'].predict(model_dict['lgbm_Dataset']['input_data']['x_test'])
        
    if "valid" in predict_on:
        model_dict['prediction_validation'] = model_dict['model'].predict(model_dict['lgbm_Dataset']['input_data']['x_valid'])
    
    #check for presence of predicted vars; if not and if they're needed for plots below, predict here
    if any([actual_v_pred_test==True, caliberation_plot_test==True, pr_curve_test==True, roc_curve_test==True]):
        if 'prediction_test' not in model_dict:
            model_dict['prediction_test'] = model_dict['model'].predict(model_dict['lgbm_Dataset']['input_data']['x_test'])
    if any([actual_v_pred_valid==True, caliberation_plot_valid==True, pr_curve_valid==True, roc_curve_valid==True]):
        if 'prediction_validation' not in model_dict:
            model_dict['prediction_validation'] = model_dict['model'].predict(model_dict['lgbm_Dataset']['input_data']['x_valid'])
        
    if caliberation_plot_test:
        caliberation_plot(model_dict, on='test')
        
    if caliberation_plot_valid:
        caliberation_plot(model_dict, on='valid')
    
    if pr_curve_test:
        pr_curve_plot(model_dict, on='test')
        
    if pr_curve_valid:
        pr_curve_plot(model_dict, on='valid')
    
    if roc_curve_test:
        roc_curve_plot(model_dict, on='test')
    
    if roc_curve_valid:
        roc_curve_plot(model_dict, on='valid')
    
    if any([actual_v_pred_test==True, actual_v_pred_valid==True]):
        if variableList==[]:
            raise ValueError("error - need to pass in variable(s) to plot")
        if actual_v_pred_test:
            df_e = model_dict['lgbm_Dataset']['input_data']['x_test'].copy()
            df_e['preds'] = model_dict['prediction_test']
            df_e['actual'] = model_dict['lgbm_Dataset']['input_data']['y_test']
            plotting(df_e,variableList, qcut, plot_title='Test Data')
        if actual_v_pred_valid:
            df_e = model_dict['lgbm_Dataset']['input_data']['x_valid'].copy()
            df_e['preds'] = model_dict['prediction_validation']
            df_e['actual'] = model_dict['lgbm_Dataset']['input_data']['y_valid']
            plotting(df_e,variableList, qcut, plot_title='Validation Data')
    return model_dict 

def model_comparison(model_dict, label=[''],
                   pc_error_diff_train=False, pc_error_diff_test=False,
                   pr_curve_diff_plt_test=False, pr_curve_plt_valid=False, 
                   roc_plot_test=False, roc_plot_valid=False,
                   calib_plot_test=False, calib_plot_valid=False):
    
    if type(model_dict) is not list:
        raise KeyError('`model_dict` must be list of 2 model dictionaries')
        
    if len(model_dict) !=2:
        raise Exception('Ensure that there must be 2 model dictionaries passed in `model_dict`')
        
    if len(model_dict) != len(label):
        label=[]
        for i in range(0,len(model_dict)):
            label.append("model"+str(i))
        print("Models have been labeled as:" , label)            
    
    for i in range(len(model_dict)): 
        if any([calib_plot_test==True, pr_curve_diff_plt_test==True, roc_plot_test==True]):
            if 'prediction_test' not in model_dict[i]:
                model_dict[i]['prediction_test'] = model_dict[i]['model'].predict(model_dict[i]['lgbm_Dataset']['input_data']['x_test'])
        if any([calib_plot_valid==True, pr_curve_plt_valid==True, roc_plot_valid==True]):
            if 'prediction_validation' not in model_dict[i]:
                model_dict[i]['prediction_validation'] = model_dict[i]['model'].predict(model_dict[i]['lgbm_Dataset']['input_data']['x_valid'])
    
    if pc_error_diff_train: 
        model1 = model_dict[0]['evals_result']['train']['binary_logloss'][-1]
        model2 = model_dict[1]['evals_result']['train']['binary_logloss'][-1]
        print('Pct change in error metric on training data ('+label[1]+'/'+label[0]+' - 1): '+str("{:.4%}".format(model2/model1 - 1)))
    
    if pc_error_diff_test: 
        model1 = model_dict[0]['evals_result']['val']['binary_logloss'][-1]
        model2 = model_dict[1]['evals_result']['val']['binary_logloss'][-1]
        print('Pct change in error metric on testing data ('+label[1]+'/'+label[0]+' - 1): '+str("{:.4%}".format(model2/model1 - 1)))
    
    if pr_curve_diff_plt_test:
        pr_curve_plot(model_dict, on='test', label=label)
    
    if pr_curve_plt_valid:
        pr_curve_plot(model_dict, on='valid', label=label)
    
    if roc_plot_test:
        roc_curve_plot(model_dict, on='test', label=label)
        
    if roc_plot_valid:
        roc_curve_plot(model_dict, on='valid', label=label)
        
    if calib_plot_test:
        caliberation_plot(model_dict, on='test', label=label)
        
    if calib_plot_valid:
        caliberation_plot(model_dict, on='valid', label=label)



def pr_curve_plot(model_dict, on, label=''):
    if on=='test':
        y='y_test'
        p='prediction_test'
        text='Test data'
    elif on=='valid':
        y='y_valid'
        p='prediction_validation'
        text='Validation data'    
    plt.figure()
    if type(model_dict) is not list: #adding in capacity to loop through multiple models; if passing in a dictionary, convert it to a list
        model_dict = [model_dict]
    else:
        model_dict = model_dict
    if type(label) is not list: #same as above. if not list, make it one for purposes of looping (without breaking rest of code)
        label = [label]
    for i in range(len(model_dict)):
        precision, recall, threshold = precision_recall_curve(model_dict[i]['lgbm_Dataset']['input_data'][y], model_dict[i][p])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label = label[i]+' AUC = %0.3f' %pr_auc)
    random_rate=sum(model_dict[0]['lgbm_Dataset']['input_data'][y])/len(model_dict[0]['lgbm_Dataset']['input_data'][y]) #assumes same
    plt.title('Precision-Recall - '+text)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [random_rate, random_rate], 'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    
    
def roc_curve_plot(model_dict, on, label=''):
    if on=='test':
        y='y_test'
        p='prediction_test'
        text='Test data'
    elif on=='valid':
        y='y_valid'
        p='prediction_validation'
        text='Validation data' 
    plt.figure()
    if type(model_dict) is not list: 
        model_dict = [model_dict]
    else:
        model_dict = model_dict
    if type(label) is not list: 
        label = [label]
    for i in range(len(model_dict)):
        false_positive_rate, recall, thresholds = roc_curve(model_dict[i]['lgbm_Dataset']['input_data'][y], model_dict[i][p])
        roc_auc = auc(false_positive_rate, recall)
        plt.plot(false_positive_rate, recall, label = label[i]+' AUC = %0.3f' %roc_auc)
    plt.title('ROC AUC - '+text)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out (1-Specificity)')
    plt.show()

    
    
def caliberation_plot(model_dict, on, label=''):
    if on=='test':
        y='y_test'
        p='prediction_test'
        text='Test data'
    elif on=='valid':
        y='y_valid'
        p='prediction_validation'
        text='Validation data'
    #plt.figure()
    if type(model_dict) is not list: 
        model_dict = [model_dict]
    else:
        model_dict = model_dict
    if type(label) is not list: 
        label = [label]
    for i in range(len(model_dict)):
        fop, mpv = calibration_curve(model_dict[i]['lgbm_Dataset']['input_data'][y], model_dict[i][p], n_bins=25)
        plt.plot(mpv, fop, marker='o', label=label[i])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Calibrated Curve - '+text)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.legend()
    plt.show()

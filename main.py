from utils import explain, fairness, quality, drift, report
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import shap
import numpy as np





def main():

    # -------------------------------------------------First Page-------------------------------------------------
    

    # User Choose Model
    model_path = 'data_model/churn/lgbm.joblib' #user choose
    model_type = 'rf or xgboost or lgbm' #user choose

    # Upload Train Data
    train_data = pd.read_csv('data_model/churn/train.csv')
    target_column = 'Tenure' #user choose
    train_data = train_data.drop(target_column, axis=1)
    target_type = 'classification or regression' #user choose
    target_type = 'regression'
    if target_type == 'classification':
        #user input two values
        target_values = [0, 1]
    else:
        target_values = []

    # Upload Test Data
    test_data = pd.read_csv('data_model/churn/test.csv')

    # Choose Target Column
    feature_names = list(train_data.columns)
    #print(feature_names)
    
    sample = test_data.iloc[0].values


    # 儲存解釋結果
    explanation_results = None
    quality_metrics = None
    fairness_metrics = None
    drift_metrics = None



    # -------------------------------------------------Second Page-------------------------------------------------



    # User Choose Explain Method
    lime = True
    shap = False
    quality_check = True
    fairness_check = True
    drift_check = True

    # Explain
    if lime:
        # 初始化 LIME 解釋器
        
        # 獲取 LIME 解釋
        explaination_metrics = explain.explain_prediction_lime(
            model_path=model_path,
            sample=sample,
            train_data=train_data,
            feature_names=feature_names,
            target_values=target_values,
            target_type=target_type
        )
        
    elif shap:
        # 獲取 SHAP 解釋
        explaination_metrics = explain.explain_prediction_shap(
            model_path=model_path,
            sample=sample,
            feature_names=feature_names
        )


    # Quality Check
    if quality_check:
        # 評估模型品質
        quality_metrics = quality.assess_model_quality(
            model_path=model_path,
            test_data=test_data.drop(columns=[target_column]), 
            test_labels=test_data[target_column],
            feature_names=feature_names,
            target_type=target_type
        )
        
        if target_type == 'classification':
            # 獲取混淆矩陣
            quality_metrics['confusion_matrix'] = quality.get_confusion_matrix(
                model_path=model_path,
                test_data=test_data.drop(columns=[target_column]),
                test_labels=test_data[target_column]
            )


    # Fairness Check
    if fairness_check:
        # 評估模型公平性
        protected_attributes = ['Age', 'Gender']  # 可以加入多個保護屬性(user choose column)
        fairness_metrics = fairness.assess_fairness(
            model_path=model_path,
            test_data=test_data,
            protected_attributes=protected_attributes,
            target_column=target_column,
            target_type=target_type
        )
        
        # 獲取群組指標
        fairness_metrics['group_metrics'] = fairness.get_group_metrics(
            model_path=model_path,
            test_data=test_data,
            protected_attributes=protected_attributes,
            target_column=target_column,
            target_type=target_type
        )


    # Drift Check
    if drift_check:
        # 由於 train_data 已經刪除了目標欄位，我們可以直接使用它
        reference_data = train_data
        current_data = test_data.drop(columns=[target_column])
        
        # 檢測特徵漂移
        drift_results = drift.detect_feature_drift(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # 計算詳細的漂移指標
        drift_metrics = drift.calculate_drift_metrics(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # 生成漂移報告
        drift_report = drift.generate_drift_report(
            drift_results=drift_results,
            metrics=drift_metrics
        )
        drift_metrics['report'] = drift_report



    # -------------------------------------------------Third Page-------------------------------------------------



    # Generate Form
    explanation_result, fairness_result, quality_result, drift_result, improve_result = report.generate_report(
        model_type=model_type,
        explanation_results=explaination_metrics,
        fairness_metrics=fairness_metrics,
        quality_metrics=quality_metrics,
        drift_metrics=drift_metrics,
        target_type=target_type,
    )

    print(explanation_result, fairness_result, quality_result, drift_result, improve_result)

if __name__ == "__main__":
    main()
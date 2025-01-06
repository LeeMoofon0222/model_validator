import joblib
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import shap


def explain_prediction_lime(model_path, sample, train_data, feature_names, 
                          target_values, target_type, num_features=None):
    """
    Variables:
        model_path: 模型文件路徑
        sample: 要解釋的樣本
        train_data: 訓練數據
        feature_names: 特徵名稱列表
        target_values: 目標值類別（分類問題用）
        target_type: 'classification' 或 'regression'
        num_features: 要顯示的特徵數量，預設為全部
    """
    # 確保 sample 維度正確（去除目標變數如果存在）
    if len(sample) > len(feature_names):
        sample = sample[:-1]
    
    # 設定要顯示的特徵數量
    if num_features is None:
        num_features = len(feature_names)
    
    try:
        # 初始化 LIME 解釋器
        if target_type == 'classification':
            explainer_lime = LimeTabularExplainer(
                training_data=np.array(train_data),
                feature_names=feature_names,
                class_names=[str(v) for v in target_values],  # 確保類別名稱為字符串
                mode='classification'
            )
        else:
            explainer_lime = LimeTabularExplainer(
                training_data=np.array(train_data),
                feature_names=feature_names,
                mode='regression'
            )
        
        # 載入模型
        model = joblib.load(model_path)
        
        # 獲取解釋
        if target_type == 'classification':
            explanation = explainer_lime.explain_instance(
                sample,
                model.predict_proba,
                num_features=num_features
            )
            prediction_proba = model.predict_proba(sample.reshape(1, -1))[0]
        else:
            explanation = explainer_lime.explain_instance(
                sample,
                model.predict,
                num_features=num_features
            )
            prediction_proba = model.predict(sample.reshape(1, -1))[0]
        
        # 整理結果
        feature_importance = explanation.as_list()
        
        return {
            'prediction_probability': prediction_proba,
            'feature_importance': feature_importance,
            'explanation_object': explanation
        }
        
    except Exception as e:
        print(f"解釋過程中發生錯誤: {str(e)}")
        print(f"Sample 形狀: {sample.shape}")
        print(f"特徵數量: {len(feature_names)}")
        print(f"訓練數據形狀: {train_data.shape}")
        raise




def explain_global_shap(model_path, test_data, feature_names, model_type):
   model = joblib.load(model_path)
   X_subset = test_data.sample(n=min(100, len(test_data)), random_state=42)
   
   explainer = shap.TreeExplainer(model) if model_type in ["Random Forest", "XGBoost"] else shap.KernelExplainer(model.predict_proba, shap.sample(X_subset, 50))
   
   shap_values = explainer.shap_values(X_subset)
   if isinstance(shap_values, list):
       shap_values = shap_values[1]
   
   importance_dict = {}
   mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
   mean_abs_shap = mean_abs_shap.reshape(-1)  # 確保是1D array
   
   for i, name in enumerate(feature_names):
       importance_dict[name] = float(mean_abs_shap[i])
       
   return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
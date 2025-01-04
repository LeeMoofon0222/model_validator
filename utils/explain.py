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




def explain_prediction_shap(model_path, sample, feature_names):
    """
    使用 SHAP 解釋預測結果
    """
    try:
        # 載入模型
        model = joblib.load(model_path)
        
        # 確保 sample 維度正確
        if len(sample) > len(feature_names):
            sample = sample[:-1]
        
        # 創建包含特徵名稱的 DataFrame
        sample_df = pd.DataFrame([sample], columns=feature_names)
        
        # 初始化 SHAP 解釋器
        explainer_shap = shap.TreeExplainer(model)
        
        # 計算 SHAP 值
        shap_values = explainer_shap.shap_values(sample_df)
        
        # 對於二分類問題，處理 SHAP 值
        if isinstance(shap_values, list):
            # 取得正類的 SHAP 值（通常是類別 1）
            class_1_shap_values = shap_values[1][0]  # [0] 為第一個（唯一）樣本
        else:
            class_1_shap_values = shap_values[0]
            
        # 建立特徵重要性字典
        feature_importance = {}
        for feat_name, shap_val in zip(feature_names, class_1_shap_values):
            # 確保每個值是標量
            if isinstance(shap_val, np.ndarray):
                # 如果是數組，取第一個值（對於二分類，通常只需要一個值）
                shap_val = shap_val[0]
            feature_importance[feat_name] = float(shap_val)
        
        # 按絕對值大小排序
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        # 找出最重要的正面和負面特徵
        positive_features = [(k, v) for k, v in feature_importance.items() if v > 0]
        negative_features = [(k, v) for k, v in feature_importance.items() if v < 0]
        
        return {
            'shap_values': class_1_shap_values,
            'feature_importance': feature_importance,
            'meta': {
                'top_positive_features': positive_features[:5],  # 前5個正面影響
                'top_negative_features': negative_features[:5]   # 前5個負面影響
            }
        }
        
    except Exception as e:
        print(f"SHAP 解釋過程中發生錯誤: {str(e)}")
        print(f"Sample 形狀: {sample.shape}")
        print(f"特徵數量: {len(feature_names)}")
        print(f"模型類型: {type(model).__name__}")
        if 'shap_values' in locals():
            print(f"SHAP 值形狀: {np.array(shap_values).shape}")
        raise
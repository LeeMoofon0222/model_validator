import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def assess_model_quality(model_path, test_data, test_labels, feature_names, target_type, model_type):
    """評估模型品質指標
    
    Parameters:
    -----------
    model_path : str
        模型文件路徑
    test_data : DataFrame
        測試數據特徵
    test_labels : Series
        測試數據標籤
    feature_names : list
        特徵名稱列表
    target_type : str
        目標類型，'classification' 或 'regression'
        
    Returns:
    --------
    dict : 包含各項品質指標的字典
    """
    # 載入模型
    model = joblib.load(model_path)
    
    # 進行預測
    predictions = model.predict(test_data)
    
    # 初始化品質指標字典
    quality_metrics = {}
    
    if target_type == 'classification':
        # 計算分類指標
        quality_metrics.update({
            '準確率': accuracy_score(test_labels, predictions),
            '精確率': precision_score(test_labels, predictions, average='weighted'),
            '召回率': recall_score(test_labels, predictions, average='weighted'),
            'F1分數': f1_score(test_labels, predictions, average='weighted')
        })
        
        # 添加分類預測統計
        unique_classes = np.unique(test_labels)
        class_distribution = {
            f'類別 {cls} 比例': np.mean(predictions == cls)
            for cls in unique_classes
        }
        quality_metrics['類別分布'] = class_distribution
        
    elif target_type == 'regression':
        # 計算回歸指標
        quality_metrics.update({
            'MSE (均方誤差)': mean_squared_error(test_labels, predictions),
            'RMSE (均方根誤差)': np.sqrt(mean_squared_error(test_labels, predictions)),
            'MAE (平均絕對誤差)': mean_absolute_error(test_labels, predictions),
            'R2 (決定係數)': r2_score(test_labels, predictions)
        })
        
        # 計算調整後的R方(Adjusted R-squared)
        n = len(test_labels)  # 樣本數
        p = len(feature_names)  # 特徵數
        r2 = quality_metrics['R2 (決定係數)']
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        quality_metrics['Adjusted R2 (調整後決定係數)'] = adj_r2
        
        # 添加回歸預測統計
        quality_metrics['預測值統計'] = {
            '最小值': float(np.min(predictions)),
            '最大值': float(np.max(predictions)),
            '平均值': float(np.mean(predictions)),
            '中位數': float(np.median(predictions)),
            '標準差': float(np.std(predictions))
        }
    
    else:
        raise ValueError(f"不支持的目標類型: {target_type}，必須是 'classification' 或 'regression'")
    
    # 如果模型有特徵重要性，加入分析
    if hasattr(model, 'feature_importances_') and model_type != 'SVM':
        feature_importance = dict(zip(feature_names, 
                                    model.feature_importances_.astype(float)))
        quality_metrics['特徵重要性'] = feature_importance
    
    return quality_metrics

def get_confusion_matrix(model_path, test_data, test_labels):
    """獲取混淆矩陣"""
    model = joblib.load(model_path)
    predictions = model.predict(test_data)
    
    # 計算混淆矩陣中的值
    tp = ((predictions == 1) & (test_labels == 1)).sum()
    tn = ((predictions == 0) & (test_labels == 0)).sum()
    fp = ((predictions == 1) & (test_labels == 0)).sum()
    fn = ((predictions == 0) & (test_labels == 1)).sum()
    
    return {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn
    }
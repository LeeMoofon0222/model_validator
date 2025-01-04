import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def assess_model_quality(model_path, test_data, test_labels, feature_names):
    """評估模型品質指標"""
    # 載入模型
    model = joblib.load(model_path)
    
    # 進行預測
    predictions = model.predict(test_data)
    
    # 計算基本指標
    quality_metrics = {
        '準確率': accuracy_score(test_labels, predictions),
        '精確率': precision_score(test_labels, predictions),
        '召回率': recall_score(test_labels, predictions),
        'F1分數': f1_score(test_labels, predictions)
    }
    
    # 如果模型有特徵重要性，加入分析
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, 
                                    model.feature_importances_))
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
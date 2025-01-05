import joblib
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np
from scipy import stats

def assess_fairness(model_path, test_data, protected_attributes, target_column, target_type='classification'):
    """評估模型公平性指標，支援分類和回歸問題
    
    Parameters:
    -----------
    model_path : str
        模型文件路徑
    test_data : DataFrame
        測試數據
    protected_attributes : list
        保護屬性列表
    target_column : str
        目標變量列名
    target_type : str
        'classification' 或 'regression'
    """
    model = joblib.load(model_path)
    feature_cols = [col for col in test_data.columns if col != target_column]
    all_metrics = {}
    
    for protected_attribute in protected_attributes:
        # 檢查保護屬性的唯一值
        unique_values = test_data[protected_attribute].unique()
        if len(unique_values) < 2:
            print(f"警告: {protected_attribute} 只有一個唯一值: {unique_values}")
            continue
            
        try:
            # 獲取預測值
            predictions = model.predict(test_data[feature_cols])
            
            if target_type == 'classification':
                # 分類問題的公平性評估
                # 將標籤轉換為二元形式(如果不是的話)
                labels = test_data[target_column].astype(int)
                if len(np.unique(labels)) > 2:
                    print(f"警告: {target_column} 包含多於兩個類別，將使用One-vs-Rest策略")
                    labels = (labels == labels.mode().iloc[0]).astype(int)
                
                binary_test = BinaryLabelDataset(
                    df=test_data.assign(**{target_column: labels}),
                    label_names=[target_column],
                    protected_attribute_names=[protected_attribute],
                    favorable_label=1,
                    unfavorable_label=0
                )
                
                pred_dataset = binary_test.copy()
                pred_dataset.labels = predictions.reshape(-1, 1)
                
                metric = ClassificationMetric(
                    binary_test,
                    pred_dataset,
                    unprivileged_groups=[{protected_attribute: test_data[protected_attribute].min()}],
                    privileged_groups=[{protected_attribute: test_data[protected_attribute].max()}]
                )
                
                fairness_metrics = {
                    '差異影響': float(metric.disparate_impact()),
                    '統計差異': float(metric.statistical_parity_difference()),
                    '機會均等差異': float(metric.equal_opportunity_difference())
                }
                
            else:  # regression
                # 回歸問題的公平性評估
                # 分割不同群組
                group_0_mask = (test_data[protected_attribute] == test_data[protected_attribute].min())
                group_1_mask = (test_data[protected_attribute] == test_data[protected_attribute].max())
                
                # 獲取每個群組的實際值和預測值
                actual_0 = test_data.loc[group_0_mask, target_column]
                actual_1 = test_data.loc[group_1_mask, target_column]
                pred_0 = predictions[group_0_mask]
                pred_1 = predictions[group_1_mask]
                
                # 計算殘差
                residuals_0 = actual_0 - pred_0
                residuals_1 = actual_1 - pred_1
                
                # 計算各種公平性指標
                fairness_metrics = {
                    # 預測值差異
                    '預測值平均差異': float(np.mean(pred_1) - np.mean(pred_0)),
                    '預測值標準差比率': float(np.std(pred_1) / np.std(pred_0)),
                    
                    # 誤差差異
                    'MSE比率': float(mean_squared_error(actual_1, pred_1) / 
                                mean_squared_error(actual_0, pred_0)),
                    'MAE比率': float(mean_absolute_error(actual_1, pred_1) / 
                               mean_absolute_error(actual_0, pred_0)),
                    
                    # 殘差差異
                    '殘差平均值差異': float(np.mean(residuals_1) - np.mean(residuals_0)),
                    '殘差標準差比率': float(np.std(residuals_1) / np.std(residuals_0)),
                    
                    # R2差異
                    'R2差異': float(r2_score(actual_1, pred_1) - r2_score(actual_0, pred_0))
                }
                
                # 添加殘差的統計檢驗（檢驗兩組殘差是否來自相同分布）
                ks_statistic, p_value = stats.ks_2samp(residuals_0, residuals_1)
                fairness_metrics.update({
                    '殘差分布KS檢驗統計量': float(ks_statistic),
                    '殘差分布KS檢驗p值': float(p_value)
                })
                
        except Exception as e:
            print(f"計算 {protected_attribute} 的公平性指標時發生錯誤: {str(e)}")
            if target_type == 'classification':
                fairness_metrics = {
                    '差異影響': np.nan,
                    '統計差異': np.nan,
                    '機會均等差異': np.nan
                }
            else:
                fairness_metrics = {
                    '預測值平均差異': np.nan,
                    '預測值標準差比率': np.nan,
                    'MSE比率': np.nan,
                    'MAE比率': np.nan,
                    '殘差平均值差異': np.nan,
                    '殘差標準差比率': np.nan,
                    'R2差異': np.nan,
                    '殘差分布KS檢驗統計量': np.nan,
                    '殘差分布KS檢驗p值': np.nan
                }
            
        all_metrics[protected_attribute] = fairness_metrics
    
    return all_metrics

def get_group_metrics(model_path, test_data, protected_attributes, target_column, target_type='classification'):
    """計算不同群組的模型表現指標，支援分類和回歸問題"""
    model = joblib.load(model_path)
    feature_cols = [col for col in test_data.columns if col != target_column]
    all_group_metrics = {}
    
    for protected_attribute in protected_attributes:
        try:
            # 獲取群組
            group_0 = test_data[test_data[protected_attribute] == test_data[protected_attribute].min()]
            group_1 = test_data[test_data[protected_attribute] == test_data[protected_attribute].max()]
            
            # 檢查群組大小
            if len(group_0) == 0 or len(group_1) == 0:
                raise ValueError(f"{protected_attribute} 群組數據不足")
            
            # 計算預測
            X_0 = group_0[feature_cols]
            X_1 = group_1[feature_cols]
            pred_0 = model.predict(X_0)
            pred_1 = model.predict(X_1)
            
            # 基本群組資訊
            metrics = {
                'group_0_size': len(group_0),
                'group_1_size': len(group_1),
                'group_0_avg_actual': float(group_0[target_column].mean()),
                'group_1_avg_actual': float(group_1[target_column].mean()),
                'group_0_avg_predicted': float(np.mean(pred_0)),
                'group_1_avg_predicted': float(np.mean(pred_1))
            }
            
            if target_type == 'classification':
                # 分類指標
                metrics.update({
                    'group_0_accuracy': float(accuracy_score(group_0[target_column], pred_0)),
                    'group_1_accuracy': float(accuracy_score(group_1[target_column], pred_1))
                })
                
                # 混淆矩陣相關指標
                cm_0 = confusion_matrix(group_0[target_column], pred_0)
                cm_1 = confusion_matrix(group_1[target_column], pred_1)
                
                for i, group in enumerate(['group_0', 'group_1']):
                    cm = cm_0 if i == 0 else cm_1
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        metrics[f'{group}_precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
                        metrics[f'{group}_recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
                        metrics[f'{group}_specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
            
            else:  # regression
                # 回歸指標
                for group_name, pred, group in [
                    ('group_0', pred_0, group_0),
                    ('group_1', pred_1, group_1)
                ]:
                    actual = group[target_column]
                    metrics.update({
                        f'{group_name}_mse': float(mean_squared_error(actual, pred)),
                        f'{group_name}_rmse': float(np.sqrt(mean_squared_error(actual, pred))),
                        f'{group_name}_mae': float(mean_absolute_error(actual, pred)),
                        f'{group_name}_r2': float(r2_score(actual, pred))
                    })
                    
                    # 添加殘差統計
                    residuals = actual - pred
                    metrics.update({
                        f'{group_name}_residuals_mean': float(np.mean(residuals)),
                        f'{group_name}_residuals_std': float(np.std(residuals)),
                        f'{group_name}_residuals_skew': float(stats.skew(residuals)),
                        f'{group_name}_residuals_kurtosis': float(stats.kurtosis(residuals))
                    })
            
            all_group_metrics[protected_attribute] = metrics
            
        except Exception as e:
            print(f"計算 {protected_attribute} 的群組指標時發生錯誤: {str(e)}")
            all_group_metrics[protected_attribute] = {
                'error': str(e),
                'group_0_size': 0,
                'group_1_size': 0
            }
    
    return all_group_metrics
import joblib
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def assess_fairness(model_path, test_data, protected_attributes, target_column):
    """評估模型公平性指標，支援多個保護屬性"""
    model = joblib.load(model_path)
    feature_cols = [col for col in test_data.columns if col != target_column]
    all_metrics = {}
    
    for protected_attribute in protected_attributes:
        # 檢查保護屬性的唯一值
        unique_values = test_data[protected_attribute].unique()
        if len(unique_values) < 2:
            print(f"警告: {protected_attribute} 只有一個唯一值: {unique_values}")
            continue
            
        # 獲取最小和最大值作為非特權和特權組
        min_val = test_data[protected_attribute].min()
        max_val = test_data[protected_attribute].max()
        
        try:
            binary_test = BinaryLabelDataset(
                df=test_data,
                label_names=[target_column],
                protected_attribute_names=[protected_attribute]
            )
            
            predictions = model.predict(test_data[feature_cols])
            pred_dataset = binary_test.copy()
            pred_dataset.labels = predictions.reshape(-1, 1)
            
            metric = ClassificationMetric(
                binary_test,
                pred_dataset,
                unprivileged_groups=[{protected_attribute: min_val}],
                privileged_groups=[{protected_attribute: max_val}]
            )
            
            fairness_metrics = {
                '差異影響': float(metric.disparate_impact()),
                '統計差異': float(metric.statistical_parity_difference()),
                '機會均等差異': float(metric.equal_opportunity_difference())
            }
        except Exception as e:
            print(f"計算 {protected_attribute} 的公平性指標時發生錯誤: {str(e)}")
            fairness_metrics = {
                '差異影響': np.nan,
                '統計差異': np.nan,
                '機會均等差異': np.nan
            }
            
        all_metrics[protected_attribute] = fairness_metrics
    
    return all_metrics

def get_group_metrics(model_path, test_data, protected_attributes, target_column):
    """計算不同群組的模型表現指標，支援多個保護屬性"""
    model = joblib.load(model_path)
    feature_cols = [col for col in test_data.columns if col != target_column]
    all_group_metrics = {}
    
    for protected_attribute in protected_attributes:
        try:
            # 獲取最小和最大值作為組別
            min_val = test_data[protected_attribute].min()
            max_val = test_data[protected_attribute].max()
            
            # 分割數據為不同群組
            group_0 = test_data[test_data[protected_attribute] == min_val]
            group_1 = test_data[test_data[protected_attribute] == max_val]
            
            # 檢查群組大小
            if len(group_0) == 0 or len(group_1) == 0:
                print(f"警告: {protected_attribute} 的某個群組沒有數據")
                raise ValueError(f"{protected_attribute} 群組數據不足")
            
            # 計算每個群組的預測
            X_0 = group_0[feature_cols]
            X_1 = group_1[feature_cols]
            pred_0 = model.predict(X_0)
            pred_1 = model.predict(X_1)
            
            metrics = {
                'group_0_size': len(group_0),
                'group_1_size': len(group_1),
                'group_0_accuracy': float(accuracy_score(group_0[target_column], pred_0)),
                'group_1_accuracy': float(accuracy_score(group_1[target_column], pred_1))
            }
            metrics['accuracy_difference'] = abs(metrics['group_0_accuracy'] - metrics['group_1_accuracy'])
            
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
            
            all_group_metrics[protected_attribute] = metrics
            
        except Exception as e:
            print(f"計算 {protected_attribute} 的群組指標時發生錯誤: {str(e)}")
            all_group_metrics[protected_attribute] = {
                'group_0_accuracy': np.nan,
                'group_1_accuracy': np.nan,
                'accuracy_difference': np.nan
            }
    
    return all_group_metrics
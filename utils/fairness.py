import joblib
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import accuracy_score

def assess_fairness(model_path, test_data, protected_attribute, target_column):
    """評估模型公平性指標"""
    # 轉換為 AIF360 數據集格式
    binary_test = BinaryLabelDataset(
        df=test_data,
        label_names=[target_column],
        protected_attribute_names=[protected_attribute]
    )
    
    # 計算公平性指標
    metric = BinaryLabelDatasetMetric(
        binary_test,
        unprivileged_groups=[{protected_attribute: 0}],
        privileged_groups=[{protected_attribute: 1}]
    )
    
    fairness_metrics = {
        '差異影響': metric.disparate_impact(),
        '統計差異': metric.statistical_parity_difference(),
        '機會均等差異': metric.equal_opportunity_difference()
    }
    
    return fairness_metrics

def get_group_metrics(model_path, test_data, protected_attribute, target_column):
    """計算不同群組的模型表現"""
    model = joblib.load(model_path)
    
    # 分割數據為不同群組
    group_0 = test_data[test_data[protected_attribute] == 0]
    group_1 = test_data[test_data[protected_attribute] == 1]
    
    # 計算每個群組的預測
    pred_0 = model.predict(group_0.drop(columns=[target_column, protected_attribute]))
    pred_1 = model.predict(group_1.drop(columns=[target_column, protected_attribute]))
    
    # 計算每個群組的準確率
    acc_0 = accuracy_score(group_0[target_column], pred_0)
    acc_1 = accuracy_score(group_1[target_column], pred_1)
    
    return {
        'group_0_accuracy': acc_0,
        'group_1_accuracy': acc_1,
        'accuracy_difference': abs(acc_0 - acc_1)
    }
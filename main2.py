from utils import explain, fairness, quality, drift
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import shap
import numpy as np

def format_explanation(explanation_results):
    """Format prediction explanation section"""
    explanation = "模型分析報告\n\n"
    explanation += "1. 單一樣本預測解釋\n"
    
    # 預測機率
    probs = explanation_results['prediction_probability']
    explanation += f"預測機率分布：[{probs[0]:.3f}, {probs[1]:.3f}]\n\n"
    
    explanation += "特徵重要性：\n"
    # Sort feature importance by absolute value
    feature_imp = explanation_results['feature_importance']
    sorted_features = sorted(feature_imp, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        explanation += f"{i}. {feature}: {importance:+.2f}%\n"
    
    return explanation

def format_fairness_analysis(fairness_metrics):
    """Format fairness analysis section"""
    fairness = "\n2. 模型公平性評估\n"
    
    # 處理每個保護特徵
    for attribute, metrics in fairness_metrics['group_metrics'].items():
        fairness += f"\n{attribute}分析\n"
        # 基本指標
        fairness += f"* 準確率差異: {metrics['accuracy_difference']*100:.2f}% "
        fairness += f"(群組0: {metrics['group_0_accuracy']*100:.2f}%, "
        fairness += f"群組1: {metrics['group_1_accuracy']*100:.2f}%)\n"
        # 群組大小
        fairness += f"* 群組樣本數 - 群組0: {metrics['group_0_size']}, 群組1: {metrics['group_1_size']}\n"
        # 其他性能指標
        fairness += f"* 精確率 - 群組0: {metrics['group_0_precision']*100:.2f}%, 群組1: {metrics['group_1_precision']*100:.2f}%\n"
        fairness += f"* 召回率 - 群組0: {metrics['group_0_recall']*100:.2f}%, 群組1: {metrics['group_1_recall']*100:.2f}%\n"
    
    return fairness

def format_quality_metrics(quality_metrics):
    """Format model performance summary section"""
    summary = "\n3. 模型性能摘要\n"
    
    # 主要性能指標
    summary += "基礎指標：\n"
    summary += f"* 整體準確率: {quality_metrics['準確率']*100:.2f}%\n"
    summary += f"* 精確率: {quality_metrics['精確率']*100:.2f}%\n"
    summary += f"* 召回率: {quality_metrics['召回率']*100:.2f}%\n"
    summary += f"* F1分數: {quality_metrics['F1分數']*100:.2f}%\n\n"
    
    # 特徵重要性（Top 10）
    summary += "特徵重要性 (Top 10)：\n"
    sorted_features = sorted(quality_metrics['特徵重要性'].items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feature, importance) in enumerate(sorted_features, 1):
        summary += f"{i}. {feature}: {importance}\n"
    
    # 混淆矩陣
    conf_matrix = quality_metrics['confusion_matrix']
    summary += "\n混淆矩陣：\n"
    summary += f"* True Positive: {conf_matrix['true_positive']:,}\n"
    summary += f"* True Negative: {conf_matrix['true_negative']:,}\n"
    summary += f"* False Positive: {conf_matrix['false_positive']:,}\n"
    summary += f"* False Negative: {conf_matrix['false_negative']:,}\n"
    
    return summary

def format_drift_analysis(drift_metrics):
    """Format drift analysis section"""
    drift_text = "\n4. 數據漂移分析\n"
    
    # 顯著漂移指標 (>3%)
    significant_drifts = {k: v['mean_change'] for k, v in drift_metrics.items() 
                         if k != 'report' and 
                         isinstance(v, dict) and 
                         'mean_change' in v and 
                         abs(v['mean_change']) > 0.03}
    
    if significant_drifts:
        drift_text += "\n顯著特徵漂移 (變化>3%)：\n"
        for feature, change in sorted(significant_drifts.items(), key=lambda x: abs(x[1]), reverse=True):
            drift_text += f"* {feature}: {change*100:+.2f}%\n"
    
    # 統計顯著性漂移
    if 'report' in drift_metrics:
        report_lines = drift_metrics['report'].split('\n')
        for line in report_lines:
            if 'p-value:' in line:
                feature = line.split(':')[0].strip()
                p_value = float(line.split(':')[1].strip())
                if p_value < 0.05:
                    drift_text += f"\n* {feature} 統計顯著漂移 (p-value: {p_value:.4f})"
    
    return drift_text

def prepare_metrics_for_gpt(explanation_results, fairness_metrics, quality_metrics, drift_metrics):
    """Prepare metrics data for GPT API"""
    # 整合所有metrics
    metrics_text = ""
    
    if explanation_results:
        metrics_text += format_explanation(explanation_results)
    if fairness_metrics:
        metrics_text += format_fairness_analysis(fairness_metrics)
    if quality_metrics:
        metrics_text += format_quality_metrics(quality_metrics)
    if drift_metrics:
        metrics_text += format_drift_analysis(drift_metrics)
    
    return metrics_text

def main():
    # -------------------------------------------------First Page-------------------------------------------------
    model_path = 'data_model/diabetes/lgbm.joblib'
    train_data = pd.read_csv('data_model/diabetes/train.csv')
    train_data = train_data.drop('Diabetes_binary', axis=1)
    
    test_data = pd.read_csv('data_model/diabetes/test.csv')
    feature_names = list(train_data.columns)
    target = 'Diabetes_binary'
    sample = test_data.iloc[0].values
    
    target_type = 'classification'
    target_values = [0, 1]

    # -------------------------------------------------Second Page-------------------------------------------------
    # Analysis flags
    lime = True
    shap = False
    quality_check = True
    fairness_check = True
    drift_check = True
    
    # Perform analyses
    explanation_results = None
    if lime:
        explanation_results = explain.explain_prediction_lime(
            model_path=model_path,
            sample=sample,
            train_data=train_data,
            feature_names=feature_names,
            target_values=target_values,
            target_type=target_type
        )
    
    quality_metrics = None
    if quality_check:
        quality_metrics = quality.assess_model_quality(
            model_path=model_path,
            test_data=test_data.drop(columns=[target]), 
            test_labels=test_data[target],
            feature_names=feature_names
        )
        quality_metrics['confusion_matrix'] = quality.get_confusion_matrix(
            model_path=model_path,
            test_data=test_data.drop(columns=[target]),
            test_labels=test_data[target]
        )
    
    fairness_metrics = None
    if fairness_check:
        protected_attributes = ['Age', 'Sex']
        fairness_metrics = fairness.assess_fairness(
            model_path=model_path,
            test_data=test_data,
            protected_attributes=protected_attributes,
            target_column=target
        )
        fairness_metrics['group_metrics'] = fairness.get_group_metrics(
            model_path=model_path,
            test_data=test_data,
            protected_attributes=protected_attributes,
            target_column=target
        )
    
    drift_metrics = None
    if drift_check:
        reference_data = train_data
        current_data = test_data.drop(columns=[target])
        
        drift_results = drift.detect_feature_drift(
            reference_data=reference_data,
            current_data=current_data
        )
        
        drift_metrics = drift.calculate_drift_metrics(
            reference_data=reference_data,
            current_data=current_data
        )
        drift_metrics['report'] = drift.generate_drift_report(
            drift_results=drift_results,
            metrics=drift_metrics
        )
    
    # -------------------------------------------------Third Page-------------------------------------------------
    # Generate and print report
    metrics = prepare_metrics_for_gpt(
        explanation_results=explanation_results,
        fairness_metrics=fairness_metrics,
        quality_metrics=quality_metrics,
        drift_metrics=drift_metrics
    )
    
    print(metrics)

if __name__ == "__main__":
    main()
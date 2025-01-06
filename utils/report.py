import openai
from dotenv import load_dotenv
import os

def format_explanation(explanation_results, target_type='classification'):
    """Format prediction explanation section with safe handling of different result structures
    
    Parameters:
    -----------
    explanation_results : dict
        預測解釋結果字典
    target_type : str
        'classification' 或 'regression'
        
    Returns:
    --------
    str : 格式化的解釋文本
    """
    explanation = "Model Analysis Report\n\n"
    explanation += "1. Single Sample Prediction Explanation\n"
    
    try:
        if target_type == 'classification':
            # 安全地獲取分類問題的預測機率
            if 'prediction_probability' in explanation_results:
                probs = explanation_results['prediction_probability']
                if isinstance(probs, (list, tuple)) and len(probs) >= 2:
                    explanation += f"Prediction probability distribution: [{probs[0]:.3f}, {probs[1]:.3f}]\n\n"
                else:
                    explanation += f"Prediction probability: {probs}\n\n"
            elif 'predicted_class' in explanation_results:
                explanation += f"Predicted class: {explanation_results['predicted_class']}\n\n"
        else:  # regression
            # 安全地獲取回歸問題的預測值
            prediction = None
            if 'prediction_value' in explanation_results:
                prediction = explanation_results['prediction_value']
            elif 'prediction' in explanation_results:
                prediction = explanation_results['prediction']
            elif 'predicted_value' in explanation_results:
                prediction = explanation_results['predicted_value']
                
            if prediction is not None:
                explanation += f"Predicted value: {float(prediction):.3f}\n"
                
                # 如果有信賴區間，添加它
                if 'confidence_interval' in explanation_results:
                    ci = explanation_results['confidence_interval']
                    explanation += f"95% Confidence Interval: [{ci[0]:.3f}, {ci[1]:.3f}]\n"
                elif 'prediction_interval' in explanation_results:
                    pi = explanation_results['prediction_interval']
                    explanation += f"95% Prediction Interval: [{pi[0]:.3f}, {pi[1]:.3f}]\n"
            else:
                explanation += "Prediction value not available\n"
            
            explanation += "\n"
    
    except Exception as e:
        explanation += f"Error processing prediction results: {str(e)}\n\n"
    
    try:
        # 特徵重要性部分
        explanation += "Feature Importance:\n"
        if 'feature_importance' in explanation_results:
            feature_imp = explanation_results['feature_importance']
            # 確保特徵重要性是可排序的格式
            if isinstance(feature_imp, dict):
                sorted_features = sorted(feature_imp.items(), key=lambda x: abs(float(x[1])), reverse=True)[:5]
            elif isinstance(feature_imp, (list, tuple)):
                sorted_features = sorted(feature_imp, key=lambda x: abs(float(x[1])), reverse=True)[:5]
            else:
                sorted_features = []
                
            for i, (feature, importance) in enumerate(sorted_features, 1):
                # 確保重要性值可以被格式化為百分比
                try:
                    imp_value = float(importance)
                    explanation += f"{i}. {feature}: {imp_value:+.2f}%\n"
                except (ValueError, TypeError):
                    explanation += f"{i}. {feature}: {importance}\n"
        else:
            explanation += "Feature importance information not available\n"
            
    except Exception as e:
        explanation += f"Error processing feature importance: {str(e)}\n"
    
    return explanation

def format_fairness_analysis(fairness_metrics, target_type='classification'):
    """Format fairness analysis section with safe handling of metrics
    
    Parameters:
    -----------
    fairness_metrics : dict
        公平性指標字典
    target_type : str
        'classification' 或 'regression'
    """
    fairness = "\n2. Model Fairness Assessment\n"
    
    if not fairness_metrics or 'group_metrics' not in fairness_metrics:
        return fairness + "\nNo fairness metrics available.\n"
        
    # Process each protected attribute
    for attribute, metrics in fairness_metrics['group_metrics'].items():
        fairness += f"\n{attribute} Analysis\n"
        
        try:
            # 基本群組資訊
            group_0_size = metrics.get('group_0_size', 0)
            group_1_size = metrics.get('group_1_size', 0)
            fairness += f"* Group Sample Size - Group 0: {group_0_size}, Group 1: {group_1_size}\n"
            
            if target_type == 'classification':
                # 安全地獲取分類指標
                # 準確率差異
                if 'accuracy_difference' in metrics:
                    acc_diff = metrics['accuracy_difference'] * 100
                    group_0_acc = metrics.get('group_0_accuracy', 0) * 100
                    group_1_acc = metrics.get('group_1_accuracy', 0) * 100
                    fairness += f"* Accuracy Difference: {acc_diff:.2f}% "
                    fairness += f"(Group 0: {group_0_acc:.2f}%, "
                    fairness += f"Group 1: {group_1_acc:.2f}%)\n"
                
                # 精確率
                if all(k in metrics for k in ['group_0_precision', 'group_1_precision']):
                    g0_prec = metrics['group_0_precision'] * 100
                    g1_prec = metrics['group_1_precision'] * 100
                    fairness += f"* Precision - Group 0: {g0_prec:.2f}%, Group 1: {g1_prec:.2f}%\n"
                
                # 召回率
                if all(k in metrics for k in ['group_0_recall', 'group_1_recall']):
                    g0_rec = metrics['group_0_recall'] * 100
                    g1_rec = metrics['group_1_recall'] * 100
                    fairness += f"* Recall - Group 0: {g0_rec:.2f}%, Group 1: {g1_rec:.2f}%\n"
                
                # 特異度
                if all(k in metrics for k in ['group_0_specificity', 'group_1_specificity']):
                    g0_spec = metrics['group_0_specificity'] * 100
                    g1_spec = metrics['group_1_specificity'] * 100
                    fairness += f"* Specificity - Group 0: {g0_spec:.2f}%, Group 1: {g1_spec:.2f}%\n"
                
            else:  # regression
                # 安全地獲取回歸指標
                # RMSE
                if all(k in metrics for k in ['group_0_rmse', 'group_1_rmse']):
                    fairness += "* RMSE Comparison:\n"
                    fairness += f"  - Group 0: {metrics['group_0_rmse']:.3f}\n"
                    fairness += f"  - Group 1: {metrics['group_1_rmse']:.3f}\n"
                    if 'rmse_difference' in metrics:
                        fairness += f"  - Difference: {metrics['rmse_difference']:.3f}\n"
                
                # MAE
                if all(k in metrics for k in ['group_0_mae', 'group_1_mae']):
                    fairness += "* MAE Comparison:\n"
                    fairness += f"  - Group 0: {metrics['group_0_mae']:.3f}\n"
                    fairness += f"  - Group 1: {metrics['group_1_mae']:.3f}\n"
                    if 'mae_difference' in metrics:
                        fairness += f"  - Difference: {metrics['mae_difference']:.3f}\n"
                
                # R²
                if all(k in metrics for k in ['group_0_r2', 'group_1_r2']):
                    fairness += "* R² Score Comparison:\n"
                    fairness += f"  - Group 0: {metrics['group_0_r2']:.3f}\n"
                    fairness += f"  - Group 1: {metrics['group_1_r2']:.3f}\n"
                    if 'r2_difference' in metrics:
                        fairness += f"  - Difference: {metrics['r2_difference']:.3f}\n"
                
                # 殘差分析
                if all(k in metrics for k in ['group_0_residuals_mean', 'group_1_residuals_mean']):
                    fairness += "* Residuals Analysis:\n"
                    fairness += f"  - Mean Residuals - Group 0: {metrics['group_0_residuals_mean']:.3f}\n"
                    fairness += f"  - Mean Residuals - Group 1: {metrics['group_1_residuals_mean']:.3f}\n"
                    if all(k in metrics for k in ['group_0_residuals_std', 'group_1_residuals_std']):
                        fairness += f"  - Std Residuals - Group 0: {metrics['group_0_residuals_std']:.3f}\n"
                        fairness += f"  - Std Residuals - Group 1: {metrics['group_1_residuals_std']:.3f}\n"
                
        except Exception as e:
            fairness += f"Error processing metrics for {attribute}: {str(e)}\n"
            continue
    
    return fairness

def format_quality_metrics(quality_metrics, target_type='classification'):
    """Format model performance summary section"""
    summary = "\n3. Model Performance Summary\n"
    
    if target_type == 'classification':
        # 分類指標
        summary += "Basic Metrics:\n"
        summary += f"* Overall Accuracy: {quality_metrics['準確率']*100:.2f}%\n"
        summary += f"* Precision: {quality_metrics['精確率']*100:.2f}%\n"
        summary += f"* Recall: {quality_metrics['召回率']*100:.2f}%\n"
        summary += f"* F1 Score: {quality_metrics['F1分數']*100:.2f}%\n\n"
        
        # 混淆矩陣
        conf_matrix = quality_metrics['confusion_matrix']
        summary += "\nConfusion Matrix:\n"
        summary += f"* True Positive: {conf_matrix['true_positive']:,}\n"
        summary += f"* True Negative: {conf_matrix['true_negative']:,}\n"
        summary += f"* False Positive: {conf_matrix['false_positive']:,}\n"
        summary += f"* False Negative: {conf_matrix['false_negative']:,}\n"
    else:
        # 回歸指標
        summary += "Regression Metrics:\n"
        summary += f"* Mean Squared Error (MSE): {quality_metrics['MSE (均方誤差)']:.3f}\n"
        summary += f"* Root Mean Squared Error (RMSE): {quality_metrics['RMSE (均方根誤差)']:.3f}\n"
        summary += f"* Mean Absolute Error (MAE): {quality_metrics['MAE (平均絕對誤差)']:.3f}\n"
        summary += f"* R² Score: {quality_metrics['R2 (決定係數)']:.3f}\n"
        summary += f"* Adjusted R²: {quality_metrics['Adjusted R2 (調整後決定係數)']:.3f}\n\n"
    
    # Feature importance (Top 10)
    if '特徵重要性' in quality_metrics:
        summary += "Feature Importance (Top 10):\n"
        sorted_features = sorted(quality_metrics['特徵重要性'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            summary += f"{i}. {feature}: {importance:.4f}\n"
    
    return summary

def format_drift_analysis(drift_metrics):
    """Format drift analysis section"""
    drift_text = "\n4. Data Drift Analysis\n"
    
    # Significant drift indicators (>3%)
    significant_drifts = {k: v['mean_change'] for k, v in drift_metrics.items() 
                         if k != 'report' and 
                         isinstance(v, dict) and 
                         'mean_change' in v and 
                         abs(v['mean_change']) > 0.03}
    
    if significant_drifts:
        drift_text += "\nSignificant Feature Drift (change >3%):\n"
        for feature, change in sorted(significant_drifts.items(), key=lambda x: abs(x[1]), reverse=True):
            drift_text += f"* {feature}: {change*100:+.2f}%\n"
    
    # Statistically significant drift
    if 'report' in drift_metrics:
        report_lines = drift_metrics['report'].split('\n')
        for line in report_lines:
            if 'p-value:' in line:
                feature = line.split(':')[0].strip()
                p_value = float(line.split(':')[1].strip())
                if p_value < 0.05:
                    drift_text += f"\n* {feature} Statistically Significant Drift (p-value: {p_value:.4f})"
    
    return drift_text

def call_gpt_api(model_type, metrics_text):
    load_dotenv()
    
    """Call GPT API to generate action plan based on metrics text."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": "你是一個專業的數據分析顧問，善於提出改進機器學習模型的行動方案。並使用繁體中文回答"},
        {"role": "user", "content": f"以下是模型的解釋結果、公平性評估、性能摘要及數據漂移分析，我用的是{model_type}方法訓練，請根據這些資訊生成建議的行動方案：\n\n{metrics_text}\n\n請以條理清晰的方式輸出建議行動方案，條目清晰，並著重於提升模型公平性、性能和應對數據漂移。"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=5000
    )

    return response.choices[0].message.content.strip()

def generate_report(model_type, explanation_results, fairness_metrics, quality_metrics, drift_metrics, target_type='classification'):
    """Generate comprehensive model analysis report"""
    metrics_text = ""

    if explanation_results:
        metrics_text += format_explanation(explanation_results, target_type)
        explanation = format_explanation(explanation_results, target_type)
    if fairness_metrics:
        metrics_text += format_fairness_analysis(fairness_metrics, target_type)
        fairness = format_fairness_analysis(fairness_metrics, target_type)
    if quality_metrics:
        metrics_text += format_quality_metrics(quality_metrics, target_type)
        quality = format_quality_metrics(quality_metrics, target_type)
    if drift_metrics:
        metrics_text += format_drift_analysis(drift_metrics)
        drift = format_drift_analysis(drift_metrics)

    improve = call_gpt_api(model_type, metrics_text)
    
    return explanation, fairness, quality, drift, improve
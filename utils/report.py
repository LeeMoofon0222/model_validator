# import openai

# def format_explanation(explanation_results):
#     """Format prediction explanation section"""
#     explanation = "模型分析報告\n\n"
#     explanation += "1. 單一樣本預測解釋\n"
    
#     # 預測機率
#     probs = explanation_results['prediction_probability']
#     explanation += f"預測機率分布：[{probs[0]:.3f}, {probs[1]:.3f}]\n\n"
    
#     explanation += "特徵重要性：\n"
#     # Sort feature importance by absolute value
#     feature_imp = explanation_results['feature_importance']
#     sorted_features = sorted(feature_imp, key=lambda x: abs(x[1]), reverse=True)[:5]
    
#     for i, (feature, importance) in enumerate(sorted_features, 1):
#         explanation += f"{i}. {feature}: {importance:+.2f}%\n"
    
#     return explanation

# def format_fairness_analysis(fairness_metrics):
#     """Format fairness analysis section"""
#     fairness = "\n2. 模型公平性評估\n"
    
#     # 處理每個保護特徵
#     for attribute, metrics in fairness_metrics['group_metrics'].items():
#         fairness += f"\n{attribute}分析\n"
#         # 基本指標
#         fairness += f"* 準確率差異: {metrics['accuracy_difference']*100:.2f}% "
#         fairness += f"(群組0: {metrics['group_0_accuracy']*100:.2f}%, "
#         fairness += f"群組1: {metrics['group_1_accuracy']*100:.2f}%)\n"
#         # 群組大小
#         fairness += f"* 群組樣本數 - 群組0: {metrics['group_0_size']}, 群組1: {metrics['group_1_size']}\n"
#         # 其他性能指標
#         fairness += f"* 精確率 - 群組0: {metrics['group_0_precision']*100:.2f}%, 群組1: {metrics['group_1_precision']*100:.2f}%\n"
#         fairness += f"* 召回率 - 群組0: {metrics['group_0_recall']*100:.2f}%, 群組1: {metrics['group_1_recall']*100:.2f}%\n"
    
#     return fairness

# def format_quality_metrics(quality_metrics):
#     """Format model performance summary section"""
#     summary = "\n3. 模型性能摘要\n"
    
#     # 主要性能指標
#     summary += "基礎指標：\n"
#     summary += f"* 整體準確率: {quality_metrics['準確率']*100:.2f}%\n"
#     summary += f"* 精確率: {quality_metrics['精確率']*100:.2f}%\n"
#     summary += f"* 召回率: {quality_metrics['召回率']*100:.2f}%\n"
#     summary += f"* F1分數: {quality_metrics['F1分數']*100:.2f}%\n\n"
    
#     # 特徵重要性（Top 10）
#     summary += "特徵重要性 (Top 10)：\n"
#     sorted_features = sorted(quality_metrics['特徵重要性'].items(), key=lambda x: x[1], reverse=True)[:10]
#     for i, (feature, importance) in enumerate(sorted_features, 1):
#         summary += f"{i}. {feature}: {importance}\n"
    
#     # 混淆矩陣
#     conf_matrix = quality_metrics['confusion_matrix']
#     summary += "\n混淆矩陣：\n"
#     summary += f"* True Positive: {conf_matrix['true_positive']:,}\n"
#     summary += f"* True Negative: {conf_matrix['true_negative']:,}\n"
#     summary += f"* False Positive: {conf_matrix['false_positive']:,}\n"
#     summary += f"* False Negative: {conf_matrix['false_negative']:,}\n"
    
#     return summary

# def format_drift_analysis(drift_metrics):
#     """Format drift analysis section"""
#     drift_text = "\n4. 數據漂移分析\n"
    
#     # 顯著漂移指標 (>3%)
#     significant_drifts = {k: v['mean_change'] for k, v in drift_metrics.items() 
#                          if k != 'report' and 
#                          isinstance(v, dict) and 
#                          'mean_change' in v and 
#                          abs(v['mean_change']) > 0.03}
    
#     if significant_drifts:
#         drift_text += "\n顯著特徵漂移 (變化>3%)：\n"
#         for feature, change in sorted(significant_drifts.items(), key=lambda x: abs(x[1]), reverse=True):
#             drift_text += f"* {feature}: {change*100:+.2f}%\n"
    
#     # 統計顯著性漂移
#     if 'report' in drift_metrics:
#         report_lines = drift_metrics['report'].split('\n')
#         for line in report_lines:
#             if 'p-value:' in line:
#                 feature = line.split(':')[0].strip()
#                 p_value = float(line.split(':')[1].strip())
#                 if p_value < 0.05:
#                     drift_text += f"\n* {feature} 統計顯著漂移 (p-value: {p_value:.4f})"
    
#     return drift_text

# def call_gpt_api(model_type, metrics_text):
#     """Call GPT API to generate action plan based on metrics text."""
#     # 配置 OpenAI client
#     client = openai.OpenAI(
#         api_key=""
#     )

#     # 定義 ChatGPT 的請求參數
#     messages = [
#         {"role": "system", "content": "你是一個專業的數據分析顧問，善於提出改進機器學習模型的行動方案。"},
#         {"role": "user", "content": f"以下是模型的解釋結果、公平性評估、性能摘要及數據漂移分析，我用的是{model_type}方法訓練，請根據這些資訊生成建議的行動方案：\n\n{metrics_text}\n\n請以條理清晰的方式輸出建議行動方案，條目清晰，並著重於提升模型公平性、性能和應對數據漂移。"}
#     ]

#     # 呼叫 GPT 模型
#     response = client.chat.completions.create(
#         model="gpt-4o",  # 更新為正確的模型名稱
#         messages=messages,
#         temperature=0.7,
#         max_tokens=5000
#     )

#     # 返回生成的文字
#     return response.choices[0].message.content.strip()

# def generate_report(model_type, explanation_results, fairness_metrics, quality_metrics, drift_metrics):
#     """Prepare metrics data for GPT API"""
#     # 整合所有metrics
#     metrics_text = ""

#     if explanation_results:
#         metrics_text += format_explanation(explanation_results)
#         explanation = format_explanation(explanation_results)
#     if fairness_metrics:
#         metrics_text += format_fairness_analysis(fairness_metrics)
#         fairness = format_fairness_analysis(fairness_metrics)
#     if quality_metrics:
#         metrics_text += format_quality_metrics(quality_metrics)
#         quality = format_quality_metrics(quality_metrics)
#     if drift_metrics:
#         metrics_text += format_drift_analysis(drift_metrics)
#         drift = format_drift_analysis(drift_metrics)

#     improve = call_gpt_api(model_type, metrics_text)

#     # metrics_text += call_gpt_api(model_type, metrics_text)
    
#     return explanation, fairness, quality, drift, improve

import openai
from dotenv import load_dotenv
import os


def format_explanation(explanation_results):
    """Format prediction explanation section"""
    explanation = "Model Analysis Report\n\n"
    explanation += "1. Single Sample Prediction Explanation\n"
    
    # Prediction probabilities
    probs = explanation_results['prediction_probability']
    explanation += f"Prediction probability distribution: [{probs[0]:.3f}, {probs[1]:.3f}]\n\n"
    
    explanation += "Feature Importance:\n"
    # Sort feature importance by absolute value
    feature_imp = explanation_results['feature_importance']
    sorted_features = sorted(feature_imp, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        explanation += f"{i}. {feature}: {importance:+.2f}%\n"
    
    return explanation

def format_fairness_analysis(fairness_metrics):
    """Format fairness analysis section"""
    fairness = "\n2. Model Fairness Assessment\n"
    
    # Process each protected attribute
    for attribute, metrics in fairness_metrics['group_metrics'].items():
        fairness += f"\n{attribute} Analysis\n"
        # Basic metrics
        fairness += f"* Accuracy Difference: {metrics['accuracy_difference']*100:.2f}% "
        fairness += f"(Group 0: {metrics['group_0_accuracy']*100:.2f}%, "
        fairness += f"Group 1: {metrics['group_1_accuracy']*100:.2f}%)\n"
        # Group sizes
        fairness += f"* Group Sample Size - Group 0: {metrics['group_0_size']}, Group 1: {metrics['group_1_size']}\n"
        # Other performance metrics
        fairness += f"* Precision - Group 0: {metrics['group_0_precision']*100:.2f}%, Group 1: {metrics['group_1_precision']*100:.2f}%\n"
        fairness += f"* Recall - Group 0: {metrics['group_0_recall']*100:.2f}%, Group 1: {metrics['group_1_recall']*100:.2f}%\n"
    
    return fairness

def format_quality_metrics(quality_metrics):
    """Format model performance summary section"""
    summary = "\n3. Model Performance Summary\n"
    
    # Main performance metrics
    summary += "Basic Metrics:\n"
    summary += f"* Overall Accuracy: {quality_metrics['準確率']*100:.2f}%\n"
    summary += f"* Precision: {quality_metrics['精確率']*100:.2f}%\n"
    summary += f"* Recall: {quality_metrics['召回率']*100:.2f}%\n"
    summary += f"* F1 Score: {quality_metrics['F1分數']*100:.2f}%\n\n"
    
    # Feature importance (Top 10)
    summary += "Feature Importance (Top 10):\n"
    sorted_features = sorted(quality_metrics['特徵重要性'].items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feature, importance) in enumerate(sorted_features, 1):
        summary += f"{i}. {feature}: {importance}\n"
    
    # Confusion matrix
    conf_matrix = quality_metrics['confusion_matrix']
    summary += "\nConfusion Matrix:\n"
    summary += f"* True Positive: {conf_matrix['true_positive']:,}\n"
    summary += f"* True Negative: {conf_matrix['true_negative']:,}\n"
    summary += f"* False Positive: {conf_matrix['false_positive']:,}\n"
    summary += f"* False Negative: {conf_matrix['false_negative']:,}\n"
    
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
    # Configure OpenAI client
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define ChatGPT request parameters
    messages = [
        {"role": "system", "content": "You are a professional data analysis consultant, skilled at proposing action plans to improve machine learning models."},
        {"role": "user", "content": f"Below are the model's explanation results, fairness assessment, performance summary, and data drift analysis. I used the {model_type} method for training. Please generate recommended action plans based on this information:\n\n{metrics_text}\n\nPlease output recommended action plans in a clear and organized manner, with clear items, focusing on improving model fairness, performance, and addressing data drift."}
    ]

    # Call GPT model
    response = client.chat.completions.create(
        model="gpt-4o",  # Update to correct model name
        messages=messages,
        temperature=0.7,
        max_tokens=5000
    )

    # Return generated text
    return response.choices[0].message.content.strip()

def generate_report(model_type, explanation_results, fairness_metrics, quality_metrics, drift_metrics):
    """Prepare metrics data for GPT API"""
    # Integrate all metrics
    metrics_text = ""

    if explanation_results:
        metrics_text += format_explanation(explanation_results)
        explanation = format_explanation(explanation_results)
    if fairness_metrics:
        metrics_text += format_fairness_analysis(fairness_metrics)
        fairness = format_fairness_analysis(fairness_metrics)
    if quality_metrics:
        metrics_text += format_quality_metrics(quality_metrics)
        quality = format_quality_metrics(quality_metrics)
    if drift_metrics:
        metrics_text += format_drift_analysis(drift_metrics)
        drift = format_drift_analysis(drift_metrics)

    improve = call_gpt_api(model_type, metrics_text)

    # metrics_text += call_gpt_api(model_type, metrics_text)
    
    return explanation, fairness, quality, drift, improve
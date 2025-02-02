import streamlit as st
import pandas as pd
import joblib
from utils import explain, fairness, quality, drift, report
import os

st.set_page_config(
    page_title="Model Analysis Dashboard",
    layout="wide",
)

def main():
    st.title("Model Analysis Dashboard")
    
    # Sidebar configurations
    st.sidebar.header("Configuration")
    
    # Model Selection
    st.sidebar.subheader("Model Settings")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Random Forest", "XGBoost", "SVM"]
    )
    
    # Model file upload
    model_file = st.sidebar.file_uploader("Upload Model File", type=['joblib'])
    if model_file is not None:
        model_path = model_file.name
        # Save uploaded model file
        with open(model_path, 'wb') as f:
            f.write(model_file.getbuffer())
    else:
        st.sidebar.warning("Please upload a model file")
        return
    
    # Target Type Selection
    target_type = st.sidebar.selectbox(
        "Select Target Type",
        ["classification", "regression"]
    )
    
    # Classification settings
    if target_type == "classification":
        st.sidebar.subheader("Classification Settings")
        target_value_0 = st.sidebar.text_input("Class 0 Value", value="0")
        target_value_1 = st.sidebar.text_input("Class 1 Value", value="1")
        target_values = [target_value_0, target_value_1]
    else:
        target_values = []
    
    # File Upload Section
    st.header("Data Upload")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data")
        train_file = st.file_uploader("Upload Training Data", type=['csv'])
        
    with col2:
        st.subheader("Test Data")
        test_file = st.file_uploader("Upload Test Data", type=['csv'])
    
    if train_file is not None and test_file is not None:
        try:
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            
            # Target Column Selection
            target_column = st.selectbox(
                "Select Target Column",
                train_data.columns
            )
            
            # Feature names
            feature_names = list(train_data.drop(target_column, axis=1).columns)
            
            # Analysis Options
            st.header("Analysis Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                explain_method = st.radio(
                    "Explanation Method",
                    ["None", "LIME", "SHAP"]
                )
            
            with col2:
                quality_check = st.checkbox("Quality Check", True)
                fairness_check = st.checkbox("Fairness Check", True)
                
            with col3:
                drift_check = st.checkbox("Drift Check", True)
            
            # Protected Attributes for Fairness
            protected_attributes = []
            if fairness_check:
                protected_attributes = st.multiselect(
                    "Select Protected Attributes",
                    feature_names
                )
            
            # Sample Selection for Explanation
            if explain_method != "None":
                st.subheader("Select Sample for Explanation")
                sample_index = st.number_input(
                    "Sample Index",
                    min_value=0,
                    max_value=len(test_data)-1,
                    value=0
                )
                sample = test_data.iloc[sample_index].drop(target_column).values
            else:
                sample = None
            
            # Run Analysis Button
            if st.button("Run Analysis"):
                with st.spinner("Running analysis..."):
                    try:
                        # Initialize all metrics
                        explanation_metrics = None
                        quality_metrics = None
                        fairness_metrics = None
                        drift_metrics = None
                        drift_results = None
                        
                        # Run selected analyses
                        if explain_method != "None" and sample is not None:
                            if explain_method == "LIME":
                                explanation_metrics = explain.explain_prediction_lime(
                                    model_path=model_path,
                                    sample=sample,
                                    train_data=train_data.drop(target_column, axis=1),
                                    feature_names=feature_names,
                                    target_values=target_values,
                                    target_type=target_type
                                )
                            elif explain_method == "SHAP":
                                explanation_metrics = explain.explain_global_shap(
                                    model_path=model_path,
                                    test_data=test_data.drop(columns=[target_column]),
                                    feature_names=feature_names,
                                    model_type=model_type
                                )
                        
                        if quality_check:
                            quality_metrics = quality.assess_model_quality(
                                model_path=model_path,
                                test_data=test_data.drop(columns=[target_column]),
                                test_labels=test_data[target_column],
                                feature_names=feature_names,
                                target_type=target_type,
                                model_type=model_type
                            )
                            
                            if target_type == "classification":
                                quality_metrics['confusion_matrix'] = quality.get_confusion_matrix(
                                    model_path=model_path,
                                    test_data=test_data.drop(columns=[target_column]),
                                    test_labels=test_data[target_column]
                                )
                        
                        if fairness_check and protected_attributes:
                            fairness_metrics = fairness.assess_fairness(
                                model_path=model_path,
                                test_data=test_data,
                                protected_attributes=protected_attributes,
                                target_column=target_column,
                                target_type=target_type
                            )
                            
                            fairness_metrics['group_metrics'] = fairness.get_group_metrics(
                                model_path=model_path,
                                test_data=test_data,
                                protected_attributes=protected_attributes,
                                target_column=target_column,
                                target_type=target_type
                            )
                        
                        if drift_check:
                            reference_data = train_data.drop(target_column, axis=1)
                            current_data = test_data.drop(columns=[target_column])
                            
                            drift_results = drift.detect_feature_drift(
                                reference_data=reference_data,
                                current_data=current_data
                            )
                            
                            drift_metrics = drift.calculate_drift_metrics(
                                reference_data=reference_data,
                                current_data=current_data
                            )
                            
                            if drift_results is not None and drift_metrics is not None:
                                drift_report = drift.generate_drift_report(
                                    drift_results=drift_results,
                                    metrics=drift_metrics
                                )
                                drift_metrics['report'] = drift_report
                        
                        # Generate Report
                        explanation_result, fairness_result, quality_result, drift_result, improve_result = report.generate_report(
                            model_type=model_type,
                            explanation_results=explanation_metrics,
                            fairness_metrics=fairness_metrics,
                            quality_metrics=quality_metrics,
                            drift_metrics=drift_metrics,
                            target_type=target_type,
                        )
                        
                        # Display Results
                        st.header("Analyze Results")
                        
                        # Use columns to organize content
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if explanation_result:
                                if explain_method == "LIME":
                                    st.markdown("### 🎯 Local Explanations")
                                    if target_type == "classification" and "probability: [" in explanation_result:
                                        prob = [float(p) for p in explanation_result.split("probability: [")[1].split("]")[0].split()]
                                        st.metric("Predicted Probability", f"{prob[1]*100:.1f}%")
                                    elif target_type == "regression" and "Predicted value:" in explanation_result:
                                        pred_value = float(explanation_result.split("Predicted value:")[1].split("\n")[0])
                                        st.metric("Predicted Value", f"{pred_value:.3f}")
                                    
                                    st.markdown("#### Main Features")
                                    for line in explanation_result.split("\n"):
                                        if ":" in line and ("+" in line or "-" in line):
                                            feature, impact = line.split(":")
                                            impact = impact.strip()
                                            color = "green" if "+" in impact else "red"
                                            st.markdown(f"- {feature}: :{color}[{impact}]")
                                else:  # SHAP
                                    st.markdown("### 🎯 Global Feature Importance")
                                    for feature, importance in explanation_metrics.items():
                                        st.markdown(f"- {feature}: :blue[{importance:.4f}]")
                        
                        with col2:
                            if quality_result:
                                st.markdown("### 📊 Model Performance")
                                metrics = {}
                                current_section = None
                                
                                for line in quality_result.split("\n"):
                                    if not line.strip():
                                        continue
                                        
                                    if line.startswith(("Basic Metrics:", "Regression Metrics:")):
                                        current_section = "metrics"
                                        continue
                                        
                                    if line.startswith("*") and ":" in line:
                                        key, value = line.replace("*", "").split(":")
                                        key = key.strip()
                                        value = value.strip()
                                        
                                        try:
                                            if "%" in value:
                                                value = float(value.replace("%", ""))
                                            else:
                                                value = float(value)
                                            metrics[key] = value
                                        except ValueError:
                                            continue
                                
                                # Display metrics based on target type
                                if target_type == "classification":
                                    cols = st.columns(4)
                                    metrics_mapping = {
                                        'Overall Accuracy': 'Accuracy',
                                        'F1 Score': 'F1 Score',
                                        'Precision': 'Precision',
                                        'Recall': 'Recall'
                                    }
                                    for i, (metric_name, display_name) in enumerate(metrics_mapping.items()):
                                        if metric_name in metrics:
                                            cols[i].metric(display_name, f"{metrics[metric_name]:.1f}%")
                                else:  # regression
                                    cols = st.columns(3)
                                    metrics_mapping = {
                                        'Root Mean Squared Error (RMSE)': 'RMSE',
                                        'Mean Absolute Error (MAE)': 'MAE',
                                        'R² Score': 'R²'
                                    }
                                    for i, (metric_name, display_name) in enumerate(metrics_mapping.items()):
                                        if metric_name in metrics:
                                            cols[i].metric(display_name, f"{metrics[metric_name]:.3f}")

   
                                # # Display Feature Importance
                                if '特徵重要性' in quality_metrics:
                                    st.markdown("##### Top 10 Important Features")
                                    # Sort features by importance and get top 10
                                    sorted_features = sorted(
                                        quality_metrics['特徵重要性'].items(), 
                                        key=lambda x: x[1], 
                                        reverse=True
                                    )[:10]
                                    
                                    # Create two columns
                                    imp_col1, imp_col2 = st.columns(2)
                                    
                                    with imp_col1:
                                        # Display feature importance as a formatted list
                                        for i, (feature, importance) in enumerate(sorted_features, 1):
                                            st.markdown(
                                                f"{i}. **{feature}**: {importance:.4f}"
                                            )
                                            
                                    
                                    with imp_col2:
                                        # Confusion Matrix visualization (only for classification)
                                        if target_type == "classification" and quality_metrics and 'confusion_matrix' in quality_metrics:
                                            st.markdown("##### Confusion Matrix")
                                            cm = quality_metrics['confusion_matrix']
                                            cm_df = pd.DataFrame([
                                                [cm['true_negative'], cm['false_positive']],  # First row: TN, FP
                                                [cm['false_negative'], cm['true_positive']]   # Second row: FN, TP
                                            ],
                                            columns=['Predicted N', 'Predicted P'],
                                            index=['Actual N', 'Actual P'])
                                            
                                            st.dataframe(cm_df.style.background_gradient(cmap='Blues'))
                                else:
                                    # Confusion Matrix visualization (only for classification)
                                        if target_type == "classification" and quality_metrics and 'confusion_matrix' in quality_metrics:
                                            st.markdown("##### Confusion Matrix")
                                            cm = quality_metrics['confusion_matrix']
                                            cm_df = pd.DataFrame([
                                                [cm['true_negative'], cm['false_positive']],  # First row: TN, FP
                                                [cm['false_negative'], cm['true_positive']]   # Second row: FN, TP
                                            ],
                                            columns=['Predicted N', 'Predicted P'],
                                            index=['Actual N', 'Actual P'])
                                            
                                            st.dataframe(cm_df.style.background_gradient(cmap='Blues'))
                        
                        
                        if fairness_result and fairness_metrics and 'group_metrics' in fairness_metrics:
                            st.markdown("### ⚖️ Fairness Analysis")
                            protected_attr_names = list(fairness_metrics['group_metrics'].keys())
                            tabs = st.tabs([f"{attr} " for attr in protected_attr_names])
                            
                            for tab, attr in zip(tabs, protected_attr_names):
                                with tab:
                                    cols = st.columns(3)
                                    metrics = fairness_metrics['group_metrics'][attr]
                                    
                                    # Display metrics based on target type
                                    if target_type == "classification":
                                        if 'group_0_accuracy' in metrics:
                                            cols[0].metric("Group 0 accuracy", f"{float(metrics['group_0_accuracy'])*100:.1f}%")
                                        if 'group_0_precision' in metrics:
                                            cols[1].metric("Group 0 precision", f"{float(metrics['group_0_precision'])*100:.1f}%")
                                        if 'group_0_recall' in metrics:
                                            cols[2].metric("Group 0 recall", f"{float(metrics['group_0_recall'])*100:.1f}%")
                                    else:  # regression
                                        if 'group_0_rmse' in metrics:
                                            cols[0].metric("Group 0 RMSE", f"{float(metrics['group_0_rmse']):.3f}")
                                        if 'group_0_mae' in metrics:
                                            cols[1].metric("Group 0 MAE", f"{float(metrics['group_0_mae']):.3f}")
                                        if 'group_0_r2' in metrics:
                                            cols[2].metric("Group 0 R²", f"{float(metrics['group_0_r2']):.3f}")
                        
                        if drift_result:
                            st.markdown("### 📈 Drift Analysis")
                            
                            # 處理漂移分析文本
                            drift_lines = drift_result.split('\n')
                            significant_drifts = []
                            
                            # 解析漂移結果
                            current_section = ""
                            for line in drift_lines:
                                line = line.strip()
                                
                                # 跳過空行和標題行
                                if not line or line.startswith('4.'):
                                    continue
                                    
                                # 檢查是否是新的章節
                                if line.endswith(':'):
                                    current_section = line
                                    continue
                                
                                # 處理顯著漂移
                                if line.startswith('*'):
                                    # 移除 * 號
                                    content = line.replace('*', '').strip()
                                    
                                    # 解析特徵和變化值
                                    if ':' in content and '%' in content:
                                        feature, change = content.split(':', 1)
                                        feature = feature.strip()
                                        try:
                                            # 提取數值
                                            change_value = float(change.replace('%', '').strip())
                                            significant_drifts.append((feature, change_value))
                                        except ValueError:
                                            continue
                            
                            # 顯示顯著特徵漂移
                            if significant_drifts:
                                st.markdown("#### Significantly Drift (>3%)")
                                for feature, drift_value in significant_drifts:
                                    # 根據漂移程度使用不同顏色
                                    if abs(drift_value) > 5:
                                        st.markdown(f"- {feature}: :red[{drift_value:+.2f}%]")
                                    else:
                                        st.markdown(f"- {feature}: :orange[{drift_value:+.2f}%]")
                            else:
                                st.info("No significant feature drift was detected")

                        # 改進建議
                        if improve_result:
                            st.markdown("### 💡 Recommendation")
                            
                            # 移除 expander，直接顯示建議
                            lines = improve_result.split('\n')
                            current_section = ""
                            
                            for line in lines:
                                line = line.strip()
                                if not line:  # 跳過空行
                                    continue
                                    
                                if line.endswith(":"):  # 如果是章節標題
                                    current_section = line
                                    st.markdown(f"#### {line}")
                                else:  # 如果是內容
                                    if current_section:  # 如果屬於某個章節
                                        st.markdown(f"- {line}")
                                    else:  # 如果是獨立內容
                                        st.markdown(line)
                    
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"Error reading data files: {str(e)}")
            st.exception(e)
    
    # Cleanup: Delete temporary model file
    try:
        if 'model_path' in locals() and os.path.exists(model_path):
            os.remove(model_path)
    except Exception as e:
        st.warning(f"Warning: Could not remove temporary model file: {str(e)}")

if __name__ == "__main__":
    main()
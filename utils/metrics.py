import joblib
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import shap


# 解釋 LIME 值
def explain_prediction(sample, num_features=10):
    # 獲取解釋
    explanation = explainer.explain_instance(
        sample, 
        model.predict_proba,
        num_features=num_features
    )
    
    # 顯示結果
    print("預測機率:", model.predict_proba(sample.reshape(1, -1))[0])
    print("\n特徵重要性:")
    for feat, imp in explanation.as_list():
        print(f"{feat}: {imp:.4f}")
    
    return explanation

# 解釋 SHAP 值
def explain_prediction_shap(sample):
    shap_values = explainer_shap.shap_values(sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    shap_values = shap_values.reshape(-1)
    feature_importance = dict(zip(feature_names[:len(shap_values)], shap_values))
    
    for name, value in feature_importance.items():
        print(f"{name}: {value:.4f}")
    
    return shap_values

# 載入模型
model = joblib.load('data_model/diabetes/rf.joblib')

# 假設你有訓練資料
X_train = pd.read_csv('data_model/diabetes/train.csv')
X_train = X_train.drop('Diabetes_binary', axis=1)
feature_names = list(X_train.columns)
sample = pd.read_csv('data_model/diabetes/test.csv')
sample = X_train.iloc[0].values

# 使用 LIME 解釋
# 創建解釋器
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=['0', '1'],  # 根據你的分類類別修改
    mode='classification'
)

# 使用 SHAP 解釋 (Tree Explainer 專門用於樹模型)
explainer_shap = shap.TreeExplainer(model)

lime_explanation = explain_prediction(sample)
shap_values = explain_prediction_shap(sample)






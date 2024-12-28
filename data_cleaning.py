import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
import joblib
import json
from datetime import datetime




def process_data(data, categorical_columns, target_column):
    """類別編碼和資料平衡"""
    for col in categorical_columns:
        if col in data.columns:
            unique_values = sorted(data[col].unique())
            encoder = OrdinalEncoder(categories=[unique_values])
            data[[col]] = encoder.fit_transform(data[[col]])
            data[col] = data[col].astype(int)
    
    grouped = data.groupby(target_column)
    return grouped.apply(lambda x: x.sample(grouped.size().min())).reset_index(drop=True)


def handle_outliers(data, numeric_columns):
    """處理離散值"""
    for column in numeric_columns:
        if column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            data = data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]
    return data


def create_sample(data, target_column, sample_size):
    """創建平衡樣本"""
    df_0 = data[data[target_column] == 0].sample(n=sample_size, random_state=42)
    df_1 = data[data[target_column] == 1].sample(n=sample_size, random_state=42)
    return pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)


def split_data(data, target_column, test_size=0.2):
    """分割訓練和測試集，並保存"""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 合併特徵和標籤
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # 保存訓練集和測試集
    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, model_type='rf', params=None):
    """
    訓練模型並返回預測結果
    model_type: 'rf'(RandomForest), 'xgb'(XGBoost), 'lgbm'(LightGBM)
    """
    # 默認參數
    default_params = {
        'rf': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'xgb': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'lgbm': {
            'n_estimators': 100,
            'num_leaves': 31,
            'random_state': 42
        }
    }

    model_params = params if params else default_params[model_type]

    # 選擇模型
    models = {
        'rf': RandomForestClassifier,
        'xgb': XGBClassifier,
        'lgbm': LGBMClassifier
    }

    # 訓練模型
    model = models[model_type](**model_params)
    model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 評估
    metrics = evaluate_model(y_pred, y_test)
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics
    }


def evaluate_model(y_pred, y_true, save_path=None):
    """評估模型表現"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(f'{save_path}_roc.png')
    plt.close()
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(f'{save_path}_conf_matrix.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist()
    }



# 使用範例
if __name__ == "__main__":
    # 讀取資料
    data = pd.read_csv('diabetes/diabetes_data.csv')
    
    # 數據預處理
    # data = process_data(data, categorical_columns=['col1', 'col2'], target_column='target')
    
    # 處理異常值
    # data = handle_outliers(data, numeric_columns=['col3', 'col4'])
    
    # 創建平衡樣本
    # data = create_sample(data, target_column='target', sample_size=5000)
    
    # 分割資料
    X_train, X_test, y_train, y_test = split_data(data, target_column='Diabetes_binary')


    # 使用隨機森林
    # results = train_model(X_train, y_train, X_test, y_test, model_type='rf')
    
    # 使用XGBoost，自定義參數
    # custom_params = {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05}
    # results = train_model(X_train, y_train, X_test, y_test, model_type='xgb', params=custom_params)
    
    # 使用LightGBM
    # results = train_model(X_train, y_train, X_test, y_test, model_type='lgbm')


    # results = train_model(X_train, y_train, X_test, y_test, model_type='rf')

    # joblib.dump(results['model'], 'rf_model.joblib')

    # metrics = results['metrics']

    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"F1-Score: {metrics['f1_score']:.4f}")
    # print(f"ROC AUC: {metrics['roc_auc']:.4f}")


    loaded_model = joblib.load('diabetes_model_rf.joblib')  # 載入模型
    if loaded_model:        
        # 進行預測
        y_pred = loaded_model.predict(X_test)

        # 評估模型
        metrics = evaluate_model(y_pred, y_test)

        # 打印評估指標
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")

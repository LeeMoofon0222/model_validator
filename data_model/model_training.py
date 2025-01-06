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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



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
    訓練分類模型並返回預測結果
    model_type: 'rf'(RandomForest), 'xgb'(XGBoost), 'lgbm'(LightGBM), 'svm'(Support Vector Machine)
    """
    default_params = {
        'rf': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'xgb': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
        'lgbm': {'n_estimators': 100, 'num_leaves': 31, 'random_state': 42},
        'svm': {'kernel': 'rbf', 'probability': True, 'random_state': 42}
    }

    model_params = params if params else default_params[model_type]

    models = {
        'rf': RandomForestClassifier,
        'xgb': XGBClassifier,
        'lgbm': LGBMClassifier,
        'svm': SVC
    }

    if model_type == 'svm':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = models[model_type](**model_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if model_type != 'svm' else model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = evaluate_model(y_pred, y_test)
    
    return {'model': model, 'predictions': y_pred, 'probabilities': y_pred_proba, 'metrics': metrics}



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


def train_regression_model(X_train, y_train, X_test, y_test, model_type='rf', params=None):
    """
    訓練回歸模型並返回預測結果
    model_type: 'rf'(RandomForest), 'xgb'(XGBoost), 'lgbm'(LightGBM), 'svm'(Support Vector Regression)
    """
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.svm import SVR

    default_params = {
        'rf': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'xgb': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
        'lgbm': {'n_estimators': 100, 'num_leaves': 31, 'random_state': 42},
        'svm': {'kernel': 'rbf'}
    }

    model_params = params if params else default_params[model_type]

    models = {
        'rf': RandomForestRegressor,
        'xgb': XGBRegressor,
        'lgbm': LGBMRegressor,
        'svm': SVR
    }

    if model_type == 'svm':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = models[model_type](**model_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_regression_model(y_pred, y_test)
    
    return {'model': model, 'predictions': y_pred, 'metrics': metrics}


def evaluate_regression_model(y_pred, y_true, save_path=None):
    """評估回歸模型表現"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 繪製預測值vs實際值的散點圖
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    
    if save_path:
        plt.savefig(f'{save_path}_pred_vs_actual.png')
    plt.close()
    
    # 繪製殘差圖
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Residuals Distribution')
    
    if save_path:
        plt.savefig(f'{save_path}_residuals.png')
    plt.close()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# 使用範例
if __name__ == "__main__":
    # 讀取資料
    data = pd.read_csv('data_model/diabetes/origin.csv')
    data = data.astype(int)
    # # 數據預處理
    # data = process_data(data, categorical_columns=['col1', 'col2'], target_column='target')
    
    # # 處理異常值
    # data = handle_outliers(data, numeric_columns=['col3', 'col4'])
    
    # # 創建平衡樣本
    # data = create_sample(data, target_column='target', sample_size=5000)
    
    # 分割資料
    X_train, X_test, y_train, y_test = split_data(data, target_column='Diabetes_binery')


    # 使用隨機森林
    # results = train_model(X_train, y_train, X_test, y_test, model_type='rf')
    
    # 使用XGBoost，自定義參數
    # custom_params = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05}
    # results = train_model(X_train, y_train, X_test, y_test, model_type='xgb', params=custom_params)
    
    # 使用svm
    results = train_model(X_train, y_train, X_test, y_test, model_type='svm')



    joblib.dump(results['model'], 'svm.joblib')

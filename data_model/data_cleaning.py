import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# 讀取數據
df = pd.read_csv('data_model/income/origin.csv')

# 1. 處理缺失值
df = df[~df.isin(['?']).any(axis=1)]

# 2. 定義數值特徵
numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                     'capital.loss', 'hours.per.week']

# 3. 將類別特徵轉換為二元特徵
# Native country: US vs non-US
df['is_US'] = (df['native.country'] == 'United-States').astype(int)

# Race: Black vs White (移除其他種族)
df = df[df['race'].isin(['White', 'Black'])]
df['is_White'] = (df['race'] == 'White').astype(int)

# Sex: Female vs Male
df['is_Male'] = (df['sex'] == 'Male').astype(int)

# 4. 定義 Ordinal Encoding 特徵及其順序
ordinal_mappings = {
    'education': [
        'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
        'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters',
        'Prof-school', 'Doctorate'
    ],
    'occupation': [
        'Other-service', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces', 'Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial',
        'Prof-specialty'
    ],
    'marital.status': [
        'Never-married', 'Married-spouse-absent', 'Separated', 'Divorced',
        'Widowed', 'Married-civ-spouse', 'Married-AF-spouse'
    ]
}

# 5. 應用 Ordinal Encoding
for column, categories in ordinal_mappings.items():
    if column in df.columns:
        encoder = OrdinalEncoder(categories=[categories])
        df[column] = encoder.fit_transform(df[[column]])

# 6. 移除不需要的列
columns_to_drop = ['workclass', 'relationship', 'race', 'sex', 'native.country']
df = df.drop(columns_to_drop, axis=1)

# 7. 處理目標變量
df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})

# 8. 轉換所有特徵為整數類型
df = df.astype(int)

# 9. 資料平衡
# 計算兩個類別的樣本數
n_income_0 = len(df[df['income'] == 0])
n_income_1 = len(df[df['income'] == 1])
min_samples = min(n_income_0, n_income_1)

# 從每個類別中隨機抽樣相同數量的樣本
df_income_0 = df[df['income'] == 0].sample(n=min_samples, random_state=42)
df_income_1 = df[df['income'] == 1].sample(n=min_samples, random_state=42)

# 合併平衡後的數據
df_balanced = pd.concat([df_income_0, df_income_1])

# 打亂數據順序
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 顯示平衡後的數據基本信息
print("\n數據預處理和平衡後的基本信息：")
print(f"資料筆數: {len(df_balanced)}")
print(f"特徵數量: {len(df_balanced.columns)}")
print("\n特徵列表：")
print(df_balanced.columns.tolist())
print("\n各類別的樣本數：")
print(df_balanced['income'].value_counts())
print("\n前五筆資料：")
print(df_balanced.head())

# 顯示數值特徵的統計摘要
print("\n數值特徵的統計摘要：")
print(df_balanced[numerical_features].describe())

# 保存處理後的數據
df_balanced.to_csv('cleaned_data.csv', index=False)
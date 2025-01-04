import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, List, Tuple

def detect_feature_drift(reference_data: pd.DataFrame, 
                        current_data: pd.DataFrame, 
                        threshold: float = 0.05) -> Dict:
    """
    檢測特徵漂移
    
    參數:
        reference_data: 參考數據集（訓練數據）
        current_data: 當前數據集（新數據）
        threshold: 顯著性水平閾值，預設為0.05
    
    返回:
        包含每個特徵漂移檢測結果的字典
    """
    drift_results = {}
    
    for column in reference_data.columns:
        # 執行 Kolmogorov-Smirnov 檢驗
        ks_statistic, p_value = stats.ks_2samp(
            reference_data[column].values,
            current_data[column].values
        )
        
        drift_detected = p_value < threshold
        
        # 計算分佈變化的基本統計量
        ref_mean = reference_data[column].mean()
        cur_mean = current_data[column].mean()
        mean_diff = abs(ref_mean - cur_mean)
        
        ref_std = reference_data[column].std()
        cur_std = current_data[column].std()
        std_diff = abs(ref_std - cur_std)
        
        drift_results[column] = {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'ks_statistic': ks_statistic,
            'mean_difference': mean_diff,
            'std_difference': std_diff
        }
    
    return drift_results

def detect_target_drift(reference_target: np.ndarray, 
                       current_target: np.ndarray, 
                       threshold: float = 0.05) -> Dict:
    """
    檢測目標變數漂移
    
    參數:
        reference_target: 參考數據集的目標變數
        current_target: 當前數據集的目標變數
        threshold: 顯著性水平閾值
    
    返回:
        目標變數漂移檢測結果
    """
    # 對於分類問題，使用卡方檢驗
    ref_counts = np.bincount(reference_target) / len(reference_target)
    cur_counts = np.bincount(current_target) / len(current_target)
    
    chi2, p_value = stats.chisquare(cur_counts, ref_counts)
    
    drift_detected = p_value < threshold
    
    # 計算類別分佈的變化
    distribution_diff = np.abs(ref_counts - cur_counts)
    
    return {
        'drift_detected': drift_detected,
        'p_value': p_value,
        'chi2_statistic': chi2,
        'distribution_difference': distribution_diff.tolist()
    }

def calculate_drift_metrics(reference_data: pd.DataFrame, 
                          current_data: pd.DataFrame, 
                          categorical_features: List[str] = None) -> Dict:
    """
    計算更詳細的漂移指標
    
    參數:
        reference_data: 參考數據集
        current_data: 當前數據集
        categorical_features: 類別特徵的列表
    
    返回:
        詳細的漂移指標
    """
    metrics = {}
    
    if categorical_features is None:
        categorical_features = []
    
    for column in reference_data.columns:
        if column in categorical_features:
            # 對類別特徵計算分佈差異
            ref_dist = reference_data[column].value_counts(normalize=True)
            cur_dist = current_data[column].value_counts(normalize=True)
            
            # 計算 JS 散度
            js_divergence = calculate_js_divergence(ref_dist, cur_dist)
            
            metrics[column] = {
                'type': 'categorical',
                'js_divergence': js_divergence,
                'distribution_changes': (cur_dist - ref_dist).to_dict()
            }
        else:
            # 對數值特徵計算統計量變化
            metrics[column] = {
                'type': 'numerical',
                'mean_change': (current_data[column].mean() - reference_data[column].mean()) / reference_data[column].mean(),
                'std_change': (current_data[column].std() - reference_data[column].std()) / reference_data[column].std(),
                'percentile_changes': calculate_percentile_changes(reference_data[column], current_data[column])
            }
    
    return metrics

def calculate_js_divergence(p: pd.Series, q: pd.Series) -> float:
    """
    計算 Jensen-Shannon 散度
    """
    # 確保兩個分佈有相同的類別
    all_categories = set(p.index) | set(q.index)
    p = p.reindex(all_categories, fill_value=0)
    q = q.reindex(all_categories, fill_value=0)
    
    # 計算平均分佈
    m = 0.5 * (p + q)
    
    # 計算 KL 散度
    kl_p_m = np.sum(p * np.log2(p / m))
    kl_q_m = np.sum(q * np.log2(q / m))
    
    # 計算 JS 散度
    js = 0.5 * (kl_p_m + kl_q_m)
    
    return js

def calculate_percentile_changes(reference_col: pd.Series, 
                               current_col: pd.Series, 
                               percentiles: List[float] = None) -> Dict:
    """
    計算百分位數的變化
    """
    if percentiles is None:
        percentiles = [25, 50, 75]
    
    changes = {}
    for p in percentiles:
        ref_p = np.percentile(reference_col, p)
        cur_p = np.percentile(current_col, p)
        changes[f'p{p}'] = (cur_p - ref_p) / ref_p if ref_p != 0 else np.inf
    
    return changes

def generate_drift_report(drift_results: Dict, metrics: Dict) -> str:
    """
    生成漂移分析報告
    
    參數:
        drift_results: detect_feature_drift 的結果
        metrics: calculate_drift_metrics 的結果
    
    返回:
        格式化的報告字符串
    """
    report = []
    report.append("# 數據漂移分析報告")
    
    # 漂移檢測摘要
    report.append("\n## 特徵漂移檢測摘要")
    drifted_features = [f for f, r in drift_results.items() if r['drift_detected']]
    report.append(f"\n檢測到漂移的特徵數量: {len(drifted_features)}")
    
    if drifted_features:
        report.append("\n存在漂移的特徵:")
        for feature in drifted_features:
            result = drift_results[feature]
            report.append(f"\n- {feature}:")
            report.append(f"  - p-value: {result['p_value']:.4f}")
            report.append(f"  - KS 統計量: {result['ks_statistic']:.4f}")
    
    # 詳細的漂移指標
    report.append("\n## 詳細漂移指標")
    for feature, metric in metrics.items():
        report.append(f"\n### {feature}")
        if metric['type'] == 'numerical':
            report.append(f"平均值變化: {metric['mean_change']:.2%}")
            report.append(f"標準差變化: {metric['std_change']:.2%}")
            report.append("\n百分位數變化:")
            for p, change in metric['percentile_changes'].items():
                report.append(f"- {p}: {change:.2%}")
        else:
            report.append(f"JS 散度: {metric['js_divergence']:.4f}")
            report.append("\n分佈變化:")
            for category, change in metric['distribution_changes'].items():
                if abs(change) > 0.01:  # 只顯示顯著變化
                    report.append(f"- {category}: {change:+.2%}")
    
    return "\n".join(report)
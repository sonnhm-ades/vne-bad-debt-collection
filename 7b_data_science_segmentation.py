# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap

OUTPUT_DIR = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science'
REPORT_DIR = os.path.join(OUTPUT_DIR, "Reports")
SUB_DATA_DIR = os.path.join(OUTPUT_DIR, "Data")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "Models")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SUB_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)




warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def get_latest_input_csv(directory):
    main_file = os.path.join(directory, "DS_PTP_PREDICTIONS.csv")
    if not os.path.exists(directory):
        return main_file
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if (f == "DS_PTP_PREDICTIONS.csv" or (f.startswith("DS_PTP_PREDICTIONS_backup_") and f.endswith(".csv")))]
    if not files:
        return main_file
    return max(files, key=os.path.getmtime)

INPUT_CSV = get_latest_input_csv(SUB_DATA_DIR)
MODEL_PKL = os.path.join(MODEL_DIR, "ptp_xgboost_model.pkl")

def run_segmentation():
    print(f"1. Nạp dữ liệu dự báo PTP từ: {os.path.basename(INPUT_CSV)}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    with open(MODEL_PKL, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    features = model_data['features']
    X_train_sample = model_data['X_train_sample']
    
    # ------------------ K-MEANS CLUSTERING ------------------
    print("2. Chạy thuật toán phân cụm K-Means (Segmentation)...")

    # Biến phân cụm: chỉ số đòn bẩy tương đối thay cho số tuyệt đối
    cluster_features = [
        'PTP_SCORE_PERCENT',            # Xác suất trả nợ từ AI (0-100)
        'DPD',                           # Số ngày quá hạn
        'TỶ_LỆ_NỢ_TRÊN_LƯƠNG',         # DTI (đòn bẩy thực tế, thay TỔNG NỢ thô)
        'SỐ_DỰ_ÁN_NGOÀI',              # Nợ chồng chéo bên ngoài (Đổi tên từ SỐ_CHỦ_NỢ_NGOÀI)
        'TỶ_LỆ_ĐÃ_THANH_TOÁN',         # % đã thanh toán (thiện chí)
        'SỐ_NGÀY_KHÔNG_THANH_TOÁN',    # Payment recency (đóng băng dòng tiền)
        'CỜ_DI_CƯ',                     # Rủi ro địa lý (khả năng bỏ trốn về quê)
    ]
    cluster_features = [c for c in cluster_features if c in df.columns]
    print(f"   => Biến phân cụm thực tế: {cluster_features}")

    # Imputation thông minh: KH thiếu thông tin ≠ KH có giá trị = 0
    df_cluster = df[cluster_features].copy()

    # Nhóm Median: giá trị thiếu không có nghĩa là bằng 0
    for col in ['DPD', 'TỶ_LỆ_NỢ_TRÊN_LƯƠNG', 'SỐ_DỰ_ÁN_NGOÀI',
                'SỐ_NGÀY_KHÔNG_THANH_TOÁN', 'PTP_SCORE_PERCENT']:
        if col in df_cluster.columns:
            median_val = df_cluster[col].median()
            df_cluster[col] = df_cluster[col].fillna(median_val)
            print(f"   => Impute '{col}': NaN → Median={median_val:.2f}")

    # Nhóm 0 hợp lý: không có data TT = chưa thanh toán; không biết di cư = giả định ổn định
    for col_zero in ['TỶ_LỆ_ĐÃ_THANH_TOÁN', 'CỜ_DI_CƯ', 'SỐ_DỰ_ÁN_NGOÀI']:
        if col_zero in df_cluster.columns:
            df_cluster[col_zero] = df_cluster[col_zero].fillna(0)

    # Missing Indicator: "Ẩn lương" là đặc trưng hành vi riêng biệt
    # Phân biệt KH "giấu lương" vs KH "thực sự thất nghiệp"
    if 'MỨC LƯƠNG' in df.columns:
        df_cluster['CỜ_NaN_LƯƠNG'] = df['MỨC LƯƠNG'].isna().astype(int)
        print(f"   => Thêm 'CỜ_NaN_LƯƠNG' — phân biệt 'Ẩn lương' (flag=1) vs 'Đủ lương' (flag=0)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Giữ cố định n_clusters=4 theo quyết định vận hành (không tự động Elbow)
    # Lý do: Số cụm thay đổi hàng tháng gây ra rủi ro vận hành cho đội Agent
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['CỤM_HÀNH_VI'] = kmeans.fit_predict(X_scaled)
    df['CỤM_HÀNH_VI'] = 'Cụm ' + (df['CỤM_HÀNH_VI'] + 1).astype(str)
    
    # ------------------ SHAP EXPLAINER (DEAD DEBT) ------------------
    print("3. Phân tích SHAP giải mã 'Dead Debt' (Nợ Chết)...")
    # Tiêu chí: PTP < 1% và chưa từng trả nợ
    dead_debt_mask = (df['PTP_SCORE_PERCENT'] < 1.0) & (df['TARGET_CÓ_TRẢ_NỢ'] == 0)
    dead_debt_idx = df[dead_debt_mask].index.tolist()
    
    print(f"   => Tìm thấy {len(dead_debt_idx)} hồ sơ Nợ Chết (PTP < 1%).")
    
    # Lấy sample tối đa 1000 người để in log giải thích (Tính SHAP cho toàn bộ sẽ rất lâu)
    sample_size = min(1000, len(dead_debt_idx))
    if sample_size > 0:
        np.random.seed(42)
        sample_idx = np.random.choice(dead_debt_idx, sample_size, replace=False)
        X_dead = df.loc[sample_idx, features]
        
        print("   => Đang tính toán SHAP Values... (Có thể mất vài phút)")
        X_train_sample = X_train_sample.astype(float)
        X_dead = X_dead.astype(float)
        explainer = shap.TreeExplainer(model, X_train_sample, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_dead)
        
        # Xử lý cấu trúc mảng của shap_values (XGBoost Classifier đôi khi trả về list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Lấy mảng của class 1 (Positive)
            
        reasons = []
        for i in range(len(X_dead)):
            sv = shap_values[i]
            # Tìm Top 3 đặc điểm "kéo" điểm xuống cực sâu (SV có giá trị ÂM lớn nhất)
            top_indices = np.argsort(sv)[:3] 
            
            reason_strs = []
            for idx in top_indices:
                feat_name = features[idx]
                feat_val = X_dead.iloc[i][feat_name]
                if isinstance(feat_val, float):
                    feat_val_str = f"{feat_val:.2f}"
                else:
                    feat_val_str = str(feat_val)
                # Dịch ra tiếng Việt thân thiện
                reason_strs.append(f"[{feat_name} = {feat_val_str}]")
                
            reasons.append(" LÀ DO: " + " VÀ ".join(reason_strs))
            
        df_reasons = pd.DataFrame({
            'LOAN ID': df.loc[sample_idx, 'LOAN ID'],
            'PTP_SCORE_PERCENT': df.loc[sample_idx, 'PTP_SCORE_PERCENT'],
            'GIẢI THÍCH TỪ AI (SHAP)': reasons
        })
        
        reason_file = os.path.join(SUB_DATA_DIR, "DEAD_DEBT_EXPLAINED.csv")
        try:
            df_reasons.to_csv(reason_file, index=False, encoding='utf-8-sig')
            print(f"   => Đã xuất log giải thích Dead Debt tại: {reason_file}")
        except PermissionError:
            backup_reason = reason_file.replace(".csv", f"_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            print(f"\n[CẢNH BÁO] Không thể ghi đè vào '{reason_file}' do file đang được mở trong Excel.")
            print(f"            Đang lưu tạm thời vào file backup: {backup_reason}")
            df_reasons.to_csv(backup_reason, index=False, encoding='utf-8-sig')
    else:
        print("   => Không có hồ sơ nào bị xếp vào Dead Debt (<1%).")

    # ------------------ SHAP TOÀN CỤC CHO DASHBOARD ------------------
    print("4. Tính toán SHAP toàn cục cho Dashboard (Top Features)...")
    X_global_sample = df[features].sample(min(2000, len(df)), random_state=42).astype(float)
    X_train_sample = X_train_sample.astype(float)
    # Re-instantiate explainer in case we skipped it above
    explainer = shap.TreeExplainer(model, X_train_sample, feature_perturbation='interventional')
    shap_values_global = explainer.shap_values(X_global_sample)
    
    if isinstance(shap_values_global, list):
        shap_values_global = shap_values_global[1]
        
    shap_data_file = os.path.join(MODEL_DIR, "shap_data_for_dashboard.pkl")
    with open(shap_data_file, 'wb') as f:
        pickle.dump({
            'shap_values': shap_values_global,
            'X_sample': X_global_sample,
            'features': features
        }, f)
        
    # Sắp xếp lại thứ tự cột: Đưa cột CỤM_HÀNH_VI lên ngay sau cột PTP_SCORE_PERCENT
    cols = list(df.columns)
    if 'CỤM_HÀNH_VI' in cols and 'PTP_SCORE_PERCENT' in cols:
        cols.remove('CỤM_HÀNH_VI')
        idx = cols.index('PTP_SCORE_PERCENT')
        cols.insert(idx + 1, 'CỤM_HÀNH_VI')
        df = df[cols]

    # Lưu file Final có cụm
    final_csv = os.path.join(SUB_DATA_DIR, "DS_SEGMENTATION_FINAL.csv")
    try:
        df.to_csv(final_csv, index=False, encoding='utf-8-sig')
        print(f"HOÀN THÀNH TASK 7B! File dữ liệu cuối cùng lưu tại: {final_csv}")
    except PermissionError:
        backup_final = final_csv.replace(".csv", f"_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"\n[CẢNH BÁO] Không thể ghi đè vào '{final_csv}' do file đang được mở trong Excel.")
        print(f"            Đang lưu tạm thời vào file backup: {backup_final}")
        df.to_csv(backup_final, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    run_segmentation()
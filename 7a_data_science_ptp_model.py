# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import re
import pickle
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

OUTPUT_DIR   = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science'
REPORT_DIR   = os.path.join(OUTPUT_DIR, "Reports")
SUB_DATA_DIR = os.path.join(OUTPUT_DIR, "Data")
MODEL_DIR    = os.path.join(OUTPUT_DIR, "Models")
os.makedirs(REPORT_DIR,   exist_ok=True)
os.makedirs(SUB_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,    exist_ok=True)

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

FILE_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH NLP
# ─────────────────────────────────────────────────────────────
POSITIVE_WORDS = ['hứa', 'xoay', 'ok', 'ck', 'chuyển khoản', 'nghe máy',
                  'đồng ý', 'thanh toán', 'có khả năng', 'sẽ gửi']
NEGATIVE_WORDS = ['chửi', 'ko trả', 'không trả', 'gắt', 'tắt máy',
                  'không hợp tác', 'công an', 'thách', 'lừa', 'kiện']
INVALID_WORDS  = ['sai số', 'thuê bao', 'kll', 'ko liên lạc', 'ảo',
                  'nhầm', 'khóa máy', 'ko nghe', 'tạm khóa']

# ─────────────────────────────────────────────────────────────
# HÀM TÁI SỬ DỤNG TỪ 8_deep_data_scan.py
# Bảo toàn đầy đủ Giờ/Phút/Giây từ phần thập phân Excel serial
# ─────────────────────────────────────────────────────────────
def to_datetime_robust(series):
    """
    Chuyển đổi chuỗi ngày tháng hỗn hợp → datetime.
    - Excel serial (float như 46063.0 hay 43843.9634375): giữ nguyên float,
      pd.to_datetime(unit='D') tự tính đủ Giờ/Phút/Giây từ phần thập phân.
    - Chuỗi DD/MM/YYYY: parse với dayfirst=True.
    - Giá trị rác ('0', '(blank)', '0//0'...): trả về NaT.
    """
    s_numeric = pd.to_numeric(series, errors='coerce')
    is_excel_date = s_numeric.notna() & (s_numeric >= 30000) & (s_numeric <= 60000)

    excel_parsed  = pd.to_datetime(s_numeric[is_excel_date],  unit='D', origin='1899-12-30')
    string_parsed = pd.to_datetime(series[~is_excel_date], errors='coerce',
                                   dayfirst=True, format='mixed')

    result = pd.Series(index=series.index, dtype='datetime64[ns]')
    result.loc[is_excel_date]  = excel_parsed
    result.loc[~is_excel_date] = string_parsed
    return result


def categorize_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return 'TRỐNG'
    text = str(text).lower()
    if any(w in text for w in POSITIVE_WORDS): return 'TÍCH CỰC'
    if any(w in text for w in NEGATIVE_WORDS): return 'TIÊU CỰC'
    if any(w in text for w in INVALID_WORDS):  return 'SỐ ẢO/SAI SỐ'
    return 'CHƯA RÕ PHÂN LOẠI'


# ─────────────────────────────────────────────────────────────
# PIPELINE CHÍNH
# ─────────────────────────────────────────────────────────────
def run_pipeline():

    # ── BƯỚC 1: NẠP DỮ LIỆU ────────────────────────────────
    print("1. Đang nạp dữ liệu từ CLEANED.csv...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    print(f"   => {len(df):,} dòng | {len(df.columns)} cột")

    # Chuẩn hóa tên cột an toàn (phòng biến thể tên cột)
    col_aliases = {
        'LƯƠNG': 'MỨC LƯƠNG', 'THU NHẬP': 'MỨC LƯƠNG', 'INCOME': 'MỨC LƯƠNG',
        'MONTH': 'THÁNG',     'PERIOD': 'THÁNG',        'KỲ': 'THÁNG',
    }
    for old, new in col_aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            print(f"   => Chuẩn hóa cột: '{old}' → '{new}'")

    # Bắt buộc sort theo thứ tự thời gian TRƯỚC KHI groupby
    # Đảm bảo chuỗi NLP [T1, T2, T3] luôn đúng chiều lịch sử (cũ → mới)
    if 'THÁNG' in df.columns:
        df['THÁNG'] = pd.to_numeric(df['THÁNG'], errors='coerce')
        df = df.sort_values(['LOAN ID', 'THÁNG'], ascending=True).reset_index(drop=True)
        print(f"   => Đã sort theo [LOAN ID, THÁNG] tăng dần (max THÁNG = {df['THÁNG'].max()}).")
    else:
        print("   [CẢNH BÁO] Không tìm thấy cột 'THÁNG'. Thứ tự NLP phụ thuộc thứ tự hàng CSV!")

    # Ép kiểu số cho các biến định lượng cốt lõi
    numeric_cols = ['KẾT QUẢ', 'NỢ GỐC', 'DPD',
                    'SỐ TIỀN GIẢI NGÂN', 'TỔNG ĐÃ THANH TOÁN',
                    'MỨC LƯƠNG', 'TUỔI', 'SỐ KỲ ĐÃ TT']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # ── BƯỚC 2: PHÂN TÍCH NLP ───────────────────────────────
    print("2. Phân tích NLP trên cột TÌNH TRẠNG LIÊN HỆ...")
    df['TEXT_LIÊN_HỆ'] = (df.get('TÌNH TRẠNG LIÊN HỆ', '').fillna('') + ' ' +
                           df.get('TÌNH TRẠNG SMS', '').fillna(''))
    df['NLP_PHÂN_LOẠI_RAW'] = df['TEXT_LIÊN_HỆ'].apply(categorize_text)

    # ── BƯỚC 3: LONG → WIDE (2 TẦNG AGG) ───────────────────
    print("3. Tổng hợp dữ liệu theo LOAN ID (Long → Wide)...")

    # TẦNG 1 — Biến đưa vào mô hình XGBoost
    MODEL_AGG = {
        'KẾT QUẢ':              'sum',   # Target: tổng tiền các tháng
        'NỢ GỐC':               'last',  # Nợ gốc gần nhất (cột thu hồi trọng tâm)
        'MỨC LƯƠNG':            'max',   # Thu nhập không đổi → max tránh null tháng sau
        'DPD':                  'last',  # DPD mới nhất
        'TUỔI':                 'max',
        'TỔNG ĐÃ THANH TOÁN':  'max',
        'SỐ TIỀN GIẢI NGÂN':   'max',
        'SỐ KỲ ĐÃ TT':         'max',
        'PHÂN LOẠI VÙNG MIỀN': 'first',
        'NLP_PHÂN_LOẠI_RAW':   list,    # Giữ list thứ tự [T1,T2,T3...] — Recency-First
    }

    # TẦNG 2 — Metadata chiến lược: lưu vào CSV cho module 8, KHÔNG vào model
    METADATA_AGG = {
        # Nhận dạng & Vận hành — module 8F (Agent Allocation)
        'DỰ ÁN':                        'first',  # 100% non-null
        'CHI NHÁNH':                    'first',  # 100% non-null
        'VNE LAW PL 01':                'last',
        'VNE LAW PL 02':                'last',
        # Địa lý — module 8D (Geography)
        'TỈNH TẠM TRÚ':                'last',   # 75.5% non-null
        'TỈNH THƯỜNG TRÚ':             'first',  # 91.8% non-null
        # Ngày tháng — để tính biến phái sinh
        'NGÀY GIẢI NGÂN':              'first',  # Ngày bắt đầu khoản vay
        'NGÀY TT GẦN NHẤT':            'last',   # Lịch sử ngân hàng — 'last' hợp lý
        # Nợ chồng chéo — module 8C (Debt Stacking)
        'KHÁCH HÀNG NHIỀU DỰ ÁN':     'last',   # Raw string
        # Pháp lý
        'HỒ SƠ KHỞI KIỆN':            'last',   # True/False
        # Sản phẩm & phân loại
        'SẢN PHẨM':                    'first',
        'PHÂN LOẠI POS':               'first',  # HIGH/MEDIUM/LOW POS
        'SỐ LƯỢNG HỢP ĐỒNG':          'first',
        # Thông tin việc làm
        'TÌNH TRẠNG VL':               'last',
        'GIỚI TÍNH':                   'first',
        # Tài chính bổ sung
        'SỐ TIỀN THANH TOÁN HÀNG THÁNG': 'max',
        'SỐ TIỀN TT GẦN NHẤT':        'last',
        # CỰC KỲ QUAN TRỌNG:
        # 'max' thay 'last' — bỏ qua NaN các tháng thất thu, giữ mốc ngày có tiền thực tế
        # Tuyệt đối KHÔNG đưa vào model (Data Leakage)
        'NGÀY CÓ KẾT QUẢ ':           'max',
    }

    all_agg = {**MODEL_AGG, **METADATA_AGG}
    all_agg = {k: v for k, v in all_agg.items() if k in df.columns}
    df_wide = df.groupby('LOAN ID').agg(all_agg).reset_index()
    print(f"   => Wide format: {len(df_wide):,} khoản vay | {len(df_wide.columns)} cột")

    # ── NLP RECENCY-FIRST (v2) ───────────────────────────────
    # Nhãn GẦN NHẤT có trọng số cao nhất
    # Phân biệt TIÊU CỰC→TÍCH CỰC (phục hồi) vs TÍCH CỰC→TIÊU CỰC (bỏ chạy)
    def final_nlp_v2(label_list):
        if not label_list:
            return 'KHÔNG CÓ DATA LIÊN HỆ'
        last_label  = label_list[-1]
        prev_labels = label_list[:-1]
        if last_label == 'TÍCH CỰC':
            if any(l == 'TIÊU CỰC' for l in prev_labels):
                return 'PHỤC HỒI: TÍCH CỰC SAU TIÊU CỰC'
            return 'LIÊN LẠC TÍCH CỰC'
        if last_label == 'TIÊU CỰC':
            if any(l == 'TÍCH CỰC' for l in prev_labels):
                return 'CẢNH BÁO: TIÊU CỰC SAU TÍCH CỰC'
            return 'LIÊN LẠC TIÊU CỰC'
        if last_label == 'SỐ ẢO/SAI SỐ':
            return 'THẤT BẠI/SỐ ẢO'
        return 'KHÔNG CÓ DATA LIÊN HỆ'

    if 'NLP_PHÂN_LOẠI_RAW' in df_wide.columns:
        df_wide['TÌNH_TRẠNG_TƯƠNG_TÁC'] = df_wide['NLP_PHÂN_LOẠI_RAW'].apply(final_nlp_v2)
        df_wide.drop('NLP_PHÂN_LOẠI_RAW', axis=1, inplace=True, errors='ignore')

    # ── BƯỚC 4: PARSE CỘT NGÀY THÁNG ───────────────────────
    print("4. Chuẩn hóa và parse các cột ngày tháng...")
    DATE_COLS = ['NGÀY GIẢI NGÂN', 'NGÀY TT GẦN NHẤT', 'NGÀY CÓ KẾT QUẢ ']
    for dc in DATE_COLS:
        if dc in df_wide.columns:
            df_wide[dc + '_PARSED'] = to_datetime_robust(df_wide[dc])
            n_ok = df_wide[dc + '_PARSED'].notna().sum()
            print(f"   => Parse '{dc}': {n_ok:,}/{len(df_wide):,} hàng thành công")

    # MAX DATE ENGINE + UPPER BOUND FILTER
    # ─────────────────────────────────────────────────────────
    # REF_DATE = ngày hợp lệ lớn nhất trong toàn bộ dataset
    # Upper Bound Filter: chỉ xét ngày <= TODAY để loại typo năm 2099, v.v.
    # Kết quả: bất biến với cùng một file dữ liệu (khác hoàn toàn với Timestamp.now())
    # ─────────────────────────────────────────────────────────
    TODAY = pd.Timestamp.today().normalize()
    parsed_available = [c for c in
                        ['NGÀY GIẢI NGÂN_PARSED', 'NGÀY TT GẦN NHẤT_PARSED', 'NGÀY CÓ KẾT QUẢ _PARSED']
                        if c in df_wide.columns]
    if parsed_available:
        all_dates  = pd.concat([df_wide[c] for c in parsed_available])
        valid_dates = all_dates[all_dates <= TODAY]
        REF_DATE   = valid_dates.max()
        print(f"   => REF_DATE (Max Date Engine + Upper Bound ≤ {TODAY.date()}): {REF_DATE.date()}")
    else:
        REF_DATE = TODAY
        print(f"   => [CẢNH BÁO] Không parse được ngày nào. Dùng TODAY={TODAY.date()} làm REF_DATE.")

    # ── BƯỚC 5: FEATURE ENGINEERING ─────────────────────────
    print("5. Sinh các biến phái sinh chiến lược (Feature Engineering)...")

    # Target nhị phân
    df_wide['TARGET_CÓ_TRẢ_NỢ'] = (df_wide.get('KẾT QUẢ', 0) > 0).astype(int)

    # Cờ tương tác gần đây
    df_wide['CỜ_TƯƠNG_TÁC_GẦN_ĐÂY'] = df_wide['TÌNH_TRẠNG_TƯƠNG_TÁC'].apply(
        lambda x: 1 if x in ['LIÊN LẠC TÍCH CỰC', 'LIÊN LẠC TIÊU CỰC'] else 0
    )

    # Tỷ lệ đã thanh toán
    df_wide['TỶ_LỆ_ĐÃ_THANH_TOÁN'] = np.where(
        df_wide.get('SỐ TIỀN GIẢI NGÂN', 1) > 0,
        df_wide.get('TỔNG ĐÃ THANH TOÁN', 0) / df_wide.get('SỐ TIỀN GIẢI NGÂN', 1),
        0
    )

    # Phân khúc tài chính DTI (v2)
    def categorize_financial_v2(row):
        luong  = row.get('MỨC LƯƠNG', np.nan)
        no_goc = row.get('NỢ GỐC',   np.nan)
        if pd.isna(luong) or luong == 0:
            return 'Nhóm 0: Giấu/Không Lương'
        luong_cao = luong > 10_000_000
        if pd.isna(no_goc) or no_goc == 0:
            return ('Nhóm 2: Lương Cao + Không Có Nợ Gốc Rõ Ràng' if luong_cao
                    else 'Nhóm 4: Lương Thấp + Không Có Nợ Gốc Rõ Ràng')
        dti = no_goc / luong
        if dti > 6 and luong_cao:     return 'Nhóm 1a: Lương Cao + Đòn Bẩy Rất Cao (DTI>6)'
        if dti > 6 and not luong_cao: return 'Nhóm 3a: Lương Thấp + Đòn Bẩy Rất Cao (DTI>6) ⚠️'
        if dti <= 6 and luong_cao:    return 'Nhóm 1b: Lương Cao + Đòn Bẩy Hợp Lý (DTI≤6)'
        return                               'Nhóm 4: Lương Thấp + Đòn Bẩy Thấp (DTI≤6)'

    df_wide['PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH'] = df_wide.apply(categorize_financial_v2, axis=1)

    # DTI dạng số (cho histogram module 8C)
    df_wide['TỶ_LỆ_NỢ_TRÊN_LƯƠNG'] = np.where(
        (df_wide.get('MỨC LƯƠNG') > 0) & (~pd.isna(df_wide.get('MỨC LƯƠNG'))),
        df_wide.get('NỢ GỐC', np.nan) / df_wide.get('MỨC LƯƠNG', np.nan),
        np.nan
    )

    # SỐ_DỰ_ÁN_NGOÀI — v4.3: kiểm tra NaN (float) TRƯỚC khi xử lý chuỗi
    def count_external_projects(val):
        if pd.isna(val):            return 0   # NaN float — bắt trước tiên, tránh AttributeError
        s = str(val).strip()
        if s in ('0', ''):          return 0   # Không có dự án ngoài
        return s.count('_') + 1               # Dấu phân tách thực tế trong data là '_'

    if 'KHÁCH HÀNG NHIỀU DỰ ÁN' in df_wide.columns:
        df_wide['SỐ_DỰ_ÁN_NGOÀI'] = df_wide['KHÁCH HÀNG NHIỀU DỰ ÁN'].apply(count_external_projects)
        print(f"   => SỐ_DỰ_ÁN_NGOÀI: max={df_wide['SỐ_DỰ_ÁN_NGOÀI'].max()} | "
              f"có dự án ngoài: {(df_wide['SỐ_DỰ_ÁN_NGOÀI'] > 0).sum():,} hồ sơ")
    else:
        df_wide['SỐ_DỰ_ÁN_NGOÀI'] = 0

    # SỐ_NGÀY_KHÔNG_THANH_TOÁN — Payment Recency dựa trên dòng tiền thực tế gần nhất
    # max(NGÀY TT GẦN NHẤT_PARSED, NGÀY CÓ KẾT QUẢ _PARSED)
    t_nganhang = df_wide['NGÀY TT GẦN NHẤT_PARSED'] if 'NGÀY TT GẦN NHẤT_PARSED' in df_wide.columns else pd.Series(pd.NaT, index=df_wide.index)
    t_vne = df_wide['NGÀY CÓ KẾT QUẢ _PARSED'] if 'NGÀY CÓ KẾT QUẢ _PARSED' in df_wide.columns else pd.Series(pd.NaT, index=df_wide.index)
    t_gan_nhat = pd.concat([t_nganhang, t_vne], axis=1).max(axis=1)

    df_wide['SỐ_NGÀY_KHÔNG_THANH_TOÁN'] = (REF_DATE - t_gan_nhat).dt.days
    df_wide.loc[df_wide['SỐ_NGÀY_KHÔNG_THANH_TOÁN'] < 0, 'SỐ_NGÀY_KHÔNG_THANH_TOÁN'] = np.nan
    n_valid = df_wide['SỐ_NGÀY_KHÔNG_THANH_TOÁN'].notna().sum()
    print(f"   => SỐ_NGÀY_KHÔNG_THANH_TOÁN (max ngân hàng & VNE): {n_valid:,} hàng có giá trị "
          f"(REF_DATE={REF_DATE.date()})")

    # CỜ_DI_CƯ — v4: chỉ gán khi CẢ HAI tỉnh đều non-null
    # fillna('') bị cấm: '' != 'NGHỆ AN' sẽ gây nhầm CỜ_DI_CƯ=1 cho KH thiếu dữ liệu
    df_wide['CỜ_DI_CƯ'] = np.nan
    if 'TỈNH TẠM TRÚ' in df_wide.columns and 'TỈNH THƯỜNG TRÚ' in df_wide.columns:
        t1 = df_wide['TỈNH TẠM TRÚ']
        t2 = df_wide['TỈNH THƯỜNG TRÚ']
        both_known = t1.notna() & t2.notna()
        df_wide.loc[both_known, 'CỜ_DI_CƯ'] = (
            t1[both_known].str.strip() != t2[both_known].str.strip()
        ).astype(int)
        n_migrant = int(df_wide['CỜ_DI_CƯ'].eq(1).sum())
        n_stable  = int(df_wide['CỜ_DI_CƯ'].eq(0).sum())
        n_unknown = int(df_wide['CỜ_DI_CƯ'].isna().sum())
        print(f"   => CỜ_DI_CƯ: {n_migrant:,} di cư | {n_stable:,} ổn định | "
              f"{n_unknown:,} không rõ (thiếu dữ liệu tỉnh)")

    # ── BƯỚC 6: ONE-HOT ENCODING (chỉ biến model) ───────────
    print("6. Mã hóa One-Hot các biến phân loại...")
    categorical_cols = ['TÌNH_TRẠNG_TƯƠNG_TÁC', 'PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH']
    if 'PHÂN LOẠI VÙNG MIỀN' in df_wide.columns:
        categorical_cols.append('PHÂN LOẠI VÙNG MIỀN')
    
    # Tạo dummy columns dạng số nguyên 0/1
    df_dummies = pd.get_dummies(df_wide[categorical_cols], dummy_na=True, dtype=int)
    # Ghép vào df_wide (giữ lại các cột chuỗi gốc)
    df_wide = pd.concat([df_wide, df_dummies], axis=1)

    # ── BƯỚC 7: XÂY DỰNG FEATURES — TƯỜNG LỬA CHỐNG RÒ RỈ ──
    print("7. Xây dựng danh sách Features (Leakage Wall)...")

    # Cột bị cấm tuyệt đối: chứa thông tin sau sự kiện (post-event)
    LEAKAGE_COLS = {
        'LOAN ID', 'KẾT QUẢ', 'TARGET_CÓ_TRẢ_NỢ',
        'NGÀY CÓ KẾT QUẢ ',         # Ngày VNE thu tiền = pure leakage
        'NGÀY CÓ KẾT QUẢ _PARSED',
        'MÃ TÌNH TRẠNG LIÊN HỆ',    # Chứa 'PAID' = leakage
    }

    # Cột metadata: lưu CSV cho module 8, KHÔNG đưa vào model
    METADATA_COLS = {
        'DỰ ÁN', 'CHI NHÁNH', 'KHÁCH HÀNG NHIỀU DỰ ÁN', 'HỒ SƠ KHỞI KIỆN',
        'TỈNH TẠM TRÚ', 'TỈNH THƯỜNG TRÚ',
        'NGÀY GIẢI NGÂN',      'NGÀY GIẢI NGÂN_PARSED',
        'NGÀY TT GẦN NHẤT',   'NGÀY TT GẦN NHẤT_PARSED',
        'SỐ LƯỢNG HỢP ĐỒNG',  'SẢN PHẨM',
        'VNE LAW PL 01',       'VNE LAW PL 02',
        'PHÂN LOẠI POS',       'TÌNH TRẠNG VL',
        'GIỚI TÍNH',
        'SỐ TIỀN THANH TOÁN HÀNG THÁNG', 'SỐ TIỀN TT GẦN NHẤT',
    }

    # Lưới an toàn: loại bỏ mọi cột object/datetime còn sót lại
    # Đồng thời loại bỏ các cột định tính gốc (categorical_cols) khỏi features vào model
    features = [
        c for c in df_wide.columns
        if c not in LEAKAGE_COLS
        and c not in METADATA_COLS
        and c not in categorical_cols
        and df_wide[c].dtype not in ['object', 'datetime64[ns]']
    ]
    print(f"   => Features vào model: {len(features)} biến")
    print(f"   => Metadata lưu CSV: {len(METADATA_COLS.intersection(set(df_wide.columns)))} cột")

    X = df_wide[features]
    y = df_wide['TARGET_CÓ_TRẢ_NỢ']

    # ── BƯỚC 8: TRAIN XGBOOST ────────────────────────────────
    print("8. Bắt đầu Train XGBoost xử lý nhãn mất cân bằng...")

    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    if num_pos == 0:
        print("LỖI: Không có dữ liệu KẾT QUẢ > 0. Mô hình không thể học!")
        return

    spw = num_neg / num_pos
    print(f"   => Nhãn 0 (Không trả): {num_neg:,} | Nhãn 1 (Chịu trả): {num_pos:,}")
    print(f"   => Kích hoạt 'scale_pos_weight' = {spw:.2f} để cân bằng.")

    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        scale_pos_weight=spw, eval_metric='auc',
        random_state=42, missing=np.nan
    )
    try:
        model.set_params(tree_method='hist', device='cuda')
        print("   => Tăng tốc XGBoost với GPU...")
    except:
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"   => Độ chính xác mô hình (AUC-ROC): {auc_score:.4f}")

    # ── BƯỚC 9: DỰ BÁO & LƯU TRỮ ───────────────────────────
    print("9. Lưu kết quả ra file...")
    df_wide['PTP_SCORE_PERCENT'] = np.round(model.predict_proba(X)[:, 1] * 100, 2)

    # Định nghĩa thứ tự cột nghiệp vụ v4.3
    # Nhóm 1: Định danh & Dự báo AI
    primary_group = [
        'LOAN ID',
        'PTP_SCORE_PERCENT',
        'TARGET_CÓ_TRẢ_NỢ'
    ]
    # Nhóm 2: Định danh & Địa lý
    customer_group = [
        'DỰ ÁN',
        'CHI NHÁNH',
        'GIỚI TÍNH',
        'TUỔI',
        'TỈNH TẠM TRÚ',
        'TỈNH THƯỜNG TRÚ',
        'CỜ_DI_CƯ'
    ]
    # Nhóm 3: Tài chính & Hợp đồng (Ngân hàng chuyển giao)
    financial_group = [
        'NỢ GỐC',
        'SỐ TIỀN GIẢI NGÂN',
        'TỔNG ĐÃ THANH TOÁN',
        'TỶ_LỆ_ĐÃ_THANH_TOÁN',
        'NGÀY GIẢI NGÂN',
        'NGÀY TT GẦN NHẤT',
        'SỐ TIỀN TT GẦN NHẤT',
        'MỨC LƯƠNG',
        'TỶ_LỆ_NỢ_TRÊN_LƯƠNG',
        'PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH',
        'SỐ TIỀN THANH TOÁN HÀNG THÁNG',
        'SỐ KỲ ĐÃ TT',
        'SỐ LƯỢNG HỢP ĐỒNG',
        'SẢN PHẨM',
        'PHÂN LOẠI POS'
    ]
    # Nhóm 4: Hiệu suất Agent & Tương tác cuộc gọi
    interaction_group = [
        'SỐ_NGÀY_KHÔNG_THANH_TOÁN',
        'NGÀY CÓ KẾT QUẢ ',
        'KẾT QUẢ',
        'TÌNH_TRẠNG_TƯƠNG_TÁC',
        'CỜ_TƯƠNG_TÁC_GẦN_ĐÂY',
        'PHÂN LOẠI VÙNG MIỀN'
    ]
    # Nhóm 5: Pháp lý, Rủi ro ngoài & Việc làm
    risk_legal_group = [
        'SỐ_DỰ_ÁN_NGOÀI',
        'HỒ SƠ KHỞI KIỆN',
        'VNE LAW PL 01',
        'VNE LAW PL 02',
        'TÌNH TRẠNG VL',
        'KHÁCH HÀNG NHIỀU DỰ ÁN'
    ]

    main_order = primary_group + customer_group + financial_group + interaction_group + risk_legal_group
    
    # Chỉ lấy các cột thực tế tồn tại trong df_wide
    existing_main = [c for c in main_order if c in df_wide.columns]
    
    # Các cột dummy và các cột còn lại khác (loại bỏ các cột ngày đã parsed _PARSED)
    remaining_cols = [
        c for c in df_wide.columns 
        if c not in existing_main 
        and not c.endswith('_PARSED')
    ]
    
    final_cols = existing_main + remaining_cols
    df_output = df_wide[final_cols]

    output_csv = os.path.join(SUB_DATA_DIR, "DS_PTP_PREDICTIONS.csv")
    try:
        df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    except PermissionError:
        backup_csv = output_csv.replace(".csv", f"_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"\n[CẢNH BÁO] Không thể ghi đè vào '{output_csv}' do file đang được mở trong Excel.")
        print(f"            Đang lưu tạm thời vào file backup: {backup_csv}")
        df_output.to_csv(backup_csv, index=False, encoding='utf-8-sig')

    model_path = os.path.join(MODEL_DIR, "ptp_xgboost_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':          model,
            'features':       features,
            'ref_date':       REF_DATE,
            'X_train_sample': X_train.sample(min(1000, len(X_train)), random_state=42)
        }, f)

    print(f"HOÀN THÀNH TASK 7A! File lưu tại:")
    print(f"  - {output_csv}")
    print(f"  - {model_path}")
    print(f"  => Tổng cột output CSV : {len(df_output.columns)}")
    print(f"  => Features vào model  : {len(features)}")
    print(f"  => REF_DATE chốt       : {REF_DATE.date()}")


if __name__ == "__main__":
    run_pipeline()
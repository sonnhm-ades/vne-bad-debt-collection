import pandas as pd
import numpy as np
import os

CSV_FILE = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'

def to_datetime_robust(series):
    """
    Chuyển đổi chuỗi dữ liệu ngày tháng sang datetime một cách mạnh mẽ.
    Hỗ trợ cả định dạng chuỗi (DD/MM/YYYY) và số serial của Excel (ví dụ: 46063.0).
    """
    s_numeric = pd.to_numeric(series, errors='coerce')
    is_excel_date = s_numeric.notna() & (s_numeric >= 30000) & (s_numeric <= 60000)
    
    excel_parsed = pd.to_datetime(s_numeric[is_excel_date], unit='D', origin='1899-12-30')
    string_parsed = pd.to_datetime(series[~is_excel_date], errors='coerce', dayfirst=True, format='mixed')
    
    result = pd.Series(index=series.index, dtype='datetime64[ns]')
    result.loc[is_excel_date] = excel_parsed
    result.loc[~is_excel_date] = string_parsed
    return result

def deep_scan():
    print("Reading data...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    
    report = []
    report.append(f"TOTAL ROWS: {len(df)}")
    report.append(f"TOTAL COLS: {len(df.columns)}")
    report.append("\n=== COLUMN ANALYSIS (Hidden Potential) ===\n")
    
    interesting_cols = [
        # --- Nhóm thông tin cơ bản & Địa lý ---
        'SẢN PHẨM', 'GIỚI TÍNH', 'TÌNH TRẠNG VL', 'CHI NHÁNH', 'TỈNH TẠM TRÚ', 'TỈNH THƯỜNG TRÚ',
        
        # --- Nhóm Lịch sử tại Ngân hàng (Dữ liệu nền) ---
        'NGÀY GIẢI NGÂN', 'SỐ TIỀN GIẢI NGÂN', 'MỨC LƯƠNG', 'SỐ KỲ', 'SỐ KỲ ĐÃ TT',
        'SỐ TIỀN THANH TOÁN HÀNG THÁNG', 'LÃI QUÁ HẠN', 'PHÍ PHẠT',
        'NGÀY TT GẦN NHẤT', 'SỐ TIỀN TT GẦN NHẤT',
        
        # --- Nhóm Kết quả tác chiến tại VNE (Dữ liệu đích) ---
        'KẾT QUẢ', 'NGÀY CÓ KẾT QUẢ ', 'MÃ TÌNH TRẠNG LIÊN HỆ', 'HỒ SƠ KHỞI KIỆN',
        
        # --- Nhóm Quản lý danh mục & Nợ chồng chéo ---
        'DỰ ÁN', 'KHÁCH HÀNG NHIỀU DỰ ÁN', 'VNE LAW PL 01', 'VNE LAW PL 02'
    ]
    
    for col in interesting_cols:
        if col in df.columns:
            non_null = df[col].count()
            unique = df[col].nunique()
            top_vals = df[col].value_counts().head(5).to_dict()
            report.append(f"COLUMN: {col}")
            report.append(f"  - Non-null: {non_null} ({non_null/len(df)*100:.1f}%)")
            report.append(f"  - Unique values: {unique}")
            report.append(f"  - Top 5 values: {top_vals}")
            report.append("-" * 30)
            
    # ----------------- KIỂM TRA XUNG ĐỘT LOGIC DỮ LIỆU -----------------
    report.append("\n=== DATA LOGIC AUDIT (Conflict Detection) ===\n")
    
    # Chuyển đổi định dạng ngày tháng tạm thời để kiểm tra logic
    t_ngay_vne = to_datetime_robust(df['NGÀY CÓ KẾT QUẢ '])
    t_ngay_bank = to_datetime_robust(df['NGÀY TT GẦN NHẤT'])
    
    # 1. Kiểm tra lỗi ngược dòng thời gian
    nguoc_thoi_gian = df[t_ngay_vne.notna() & t_ngay_bank.notna() & (t_ngay_vne < t_ngay_bank)]
    report.append(f"1. Lỗi ngược thời gian (Ngày VNE thu < Ngày Ngân hàng thu): {len(nguoc_thoi_gian)} hồ sơ")
    if len(nguoc_thoi_gian) > 0:
        report.append(f"   - Danh sách LOAN ID lỗi: {nguoc_thoi_gian['LOAN ID'].head(10).tolist()}")
        
    # 2. Kiểm tra lỗi lệch pha giữa Tiền và Ngày của VNE
    co_tien_mat_ngay = df[(df['KẾT QUẢ'] > 0) & df['NGÀY CÓ KẾT QUẢ '].isna()]
    no_tien_co_ngay = df[((df['KẾT QUẢ'].isna()) | (df['KẾT QUẢ'] == 0)) & df['NGÀY CÓ KẾT QUẢ '].notna()]
    
    report.append(f"2. Lỗi có tiền thu về (KẾT QUẢ > 0) nhưng trống NGÀY CÓ KẾT QUẢ: {len(co_tien_mat_ngay)} hồ sơ")
    if len(co_tien_mat_ngay) > 0:
        report.append(f"   - Danh sách LOAN ID lỗi: {co_tien_mat_ngay['LOAN ID'].head(10).tolist()}")
        
    report.append(f"3. Lỗi không có tiền (KẾT QUẢ = 0) nhưng lại có NGÀY CÓ KẾT QUẢ: {len(no_tien_co_ngay)} hồ sơ")
    if len(no_tien_co_ngay) > 0:
        report.append(f"   - Danh sách LOAN ID lỗi: {no_tien_co_ngay['LOAN ID'].head(10).tolist()}")
        
    # 3. Kiểm tra lỗi định dạng ngày tháng bị lỗi (Không thể ép kiểu sang datetime)
    report.append("\n4. Kiểm tra lỗi định dạng ngày tháng (Tỷ lệ phân tích thất bại):")
    for ngay_col in ['NGÀY GIẢI NGÂN', 'NGÀY TT GẦN NHẤT', 'NGÀY CÓ KẾT QUẢ ']:
        if ngay_col in df.columns:
            tong_co_chu = df[ngay_col].notna().sum()
            parse_loi = df[df[ngay_col].notna() & to_datetime_robust(df[ngay_col]).isna()]
            report.append(f"   - Cột {ngay_col}: Lỗi định dạng {len(parse_loi)} / {tong_co_chu} hồ sơ có chữ")
            
    with open('deep_scan_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print("Deep scan complete. Saved to deep_scan_report.txt")

if __name__ == "__main__":
    deep_scan()

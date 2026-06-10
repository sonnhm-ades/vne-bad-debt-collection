import pandas as pd
import os
import sys

# Đảm bảo in được ký tự tiếng Việt trên console Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình đường dẫn file
FILE_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026.xlsx'

def data_understanding(file_path):
    """
    Hàm thực hiện các bước tìm hiểu dữ liệu cơ bản.
    """
    print(f"{'='*20} DATA UNDERSTANDING: {os.path.basename(file_path)} {'='*20}")
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file tại {file_path}")
        return

    # 1. Load dữ liệu dựa trên định dạng file
    print("\n[STEP 1] Đang tải dữ liệu...")
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in ['.xlsx', '.xls']:
            print("   (Lưu ý: File Excel lớn có thể mất 1-2 phút để tải...)")
            # Thử load sheet 'DOANH SỐ' nếu là file Excel, nếu không load sheet đầu tiên
            try:
                df = pd.read_excel(file_path, sheet_name='DOANH SỐ')
            except Exception:
                df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Lỗi khi tải file: {e}")
        return
    
    # 2. df.shape: Kích thước dữ liệu
    print("\n[STEP 2] Kích thước dữ liệu (df.shape):")
    print(f"- Số dòng: {df.shape[0]:,}")
    print(f"- Số cột: {df.shape[1]}")
    
    # 3. df.head(): Xem 5 dòng đầu tiên
    print("\n[STEP 3] 5 dòng dữ liệu đầu tiên (df.head()):")
    print(df.head())
    
    # 4. df.info(): Thông tin tổng quan (kiểu dữ liệu, bộ nhớ)
    print("\n[STEP 4] Thông tin cấu trúc dữ liệu (df.info()):")
    df.info()
    
    # 5. df.describe(): Thống kê mô tả (cho các cột số)
    print("\n[STEP 5] Thống kê mô tả (df.describe()):")
    print(df.describe())
    
    # 6. df.isnull().sum(): Kiểm tra giá trị thiếu (Null/NaN)
    print("\n[STEP 6] Kiểm tra giá trị thiếu (df.isnull().sum()):")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if not null_counts[null_counts > 0].empty else "Không có giá trị thiếu.")
    
    # 7. df.duplicated().sum(): Kiểm tra dòng trùng lặp
    print("\n[STEP 7] Kiểm tra dòng trùng lặp (df.duplicated().sum()):")
    duplicate_count = df.duplicated().sum()
    print(f"- Số dòng trùng lặp: {duplicate_count:,}")

    print(f"\n{'='*20} KẾT THÚC PHÂN TÍCH {'='*20}")

if __name__ == "__main__":
    data_understanding(FILE_PATH)

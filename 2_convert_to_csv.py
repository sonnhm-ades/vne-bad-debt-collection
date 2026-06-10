import pandas as pd
import sys

input_file = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026.xlsx'
sheet_name = 'DOANH SỐ'
output_file = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026.csv'

# Danh sách các cột cần drop
columns_to_drop = [
    'NAME', 'CMND', 'SỐ ĐIỆN THOẠI', 'MAIL',
    'TÊN CÔNG TY', 'ĐỊA CHỈ CÔNG TY', 'SĐT CÔNG TY',
    'THAM CHIẾU VỢ/CHỒNG', 'SỐ THAM CHIẾU 1', 'QUAN HỆ 1',
    'THAM CHIẾU 2', 'QUAN HỆ 2', 'THAM CHIẾU 3', 'QUAN HỆ 3',
    'THAM CHIẾU 4', 'QUAN HỆ 4', 'TÁC ĐỘNG CŨ',
    'STK BIDV', 'STK WORRI BANK', 'NƠI NHẬN THƯ', 'SĐT', 
    'ĐDPL', 'NGÀY GỬI', 'MÃ VẬN ĐƠN', 'HOTLINE'
]

print(f"Reading {input_file}...")
try:
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    print(f"Read {len(df)} rows.")

    # Drop các cột nếu tồn tại (tránh lỗi nếu thiếu cột)
    df = df.drop(columns=columns_to_drop, errors='ignore')

    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
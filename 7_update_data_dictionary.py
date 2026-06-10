import pandas as pd
import re

CSV_FILE = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'
MD_FILE = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\DATA_DICTIONARY.md'

def update_dictionary():
    # 1. Read existing dictionary
    existing_desc = {}
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Regex to find table rows
    table_pattern = re.compile(r'\|\s*\d+\s*\|\s*`([^`]+)`\s*\|[^\|]+\|[^\|]+\|[^\|]+\|\s*([^\|]+)\|')
    matches = table_pattern.findall(content)
    for col_name, desc in matches:
        existing_desc[col_name.strip()] = desc.strip()
        
    # 2. Get actual columns
    df = pd.read_csv(CSV_FILE, nrows=0)
    actual_cols = df.columns.tolist()
    
    # 3. Generate new markdown content
    new_md = f"""# TỪ ĐIỂN DỮ LIỆU — Bad Debt Portfolio (Updated)

**Ngày báo cáo**: 08/04/2026 (Updated automatically)
**Tệp dữ liệu**: TỔNG HỢP NĂM 2026 CLEANED.csv
**Tổng quan**: Theo dữ liệu thực tế với {len(actual_cols)} cột.

## Bảng Tổng Hợp Tất Cả Cột

| STT | Tên Cột | Định Nghĩa / Ý Nghĩa |
|-----|---------|---------------------|
"""
    
    for i, col in enumerate(actual_cols, 1):
        desc = existing_desc.get(col, "(Cột mới - Cần bổ sung định nghĩa)")
        new_md += f"| {i} | `{col}` | {desc} |\n"
        
    new_md += "\n---\n*Đã đồng bộ tự động 76 cột với file CLEANED.csv*\n"
    
    with open(MD_FILE, 'w', encoding='utf-8') as f:
        f.write(new_md)
    print("Updated DATA_DICTIONARY.md successfully!")

if __name__ == "__main__":
    update_dictionary()

import pandas as pd
import numpy as np

# Load original inference
input_file = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\schema_inference_output.csv'
output_csv = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\refined_schema_ai.csv'

df = pd.read_csv(input_file)

def get_refined_ai_schema(col_name, sample_val, original_type, reason, unique_count, null_count):
    col_lower = str(col_name).lower()
    sample_str = str(sample_val).lower()
    
    # 0. Check for all-null columns
    if null_count >= 20000:
        return "EMPTY", "string", "Column is entirely empty in sample", "Empty column"

    # 1. ID Keywords (Priority) - IDs must be string to avoid numerical bias
    id_kws = ['loan id', 'id', 'mã', 'cmnd', 'cccd', 'sđt', 'stk', 'account', 'contract', 'hợp đồng', 'số phiếu', 'vận đơn', 'mail']
    if any(kw in col_lower for kw in id_kws):
        return "ID", "string", "Identifier or contact info - categorical/ID (non-numeric for modeling)", ""

    # 2. NUMERIC_RATIO (Percentages, Rates)
    if any(kw in col_lower for kw in ['lãi suất', '%', 'tỷ lệ', 'ratio']):
        return "NUMERIC_RATIO", "float64", "Percentage or ratio value requiring decimal precision", ""

    # 3. NUMERIC_AMOUNT (Values of money)
    if any(kw in col_lower for kw in ['tiền', 'số dư', 'nợ', 'lãi', 'gốc', 'phí', 'lương', 'amount', 'balance', 'debt', 'principal', 'fee', 'kết quả', 'thông báo', 'mtd', 'số tiền']):
        return "NUMERIC_AMOUNT", "float64", "Monetary amount for AI processing", ""

    # 4. NUMERIC_COUNT (Discrete integers)
    if any(kw in col_lower for kw in ['tuổi', 'số kỳ', 'số lượng', 'số lần', 'count', 'lần', 'dpd']):
        # Special check for DPD - sometimes it's grouped, but usually it's an integer
        if "nhóm" in col_lower or "<" in sample_str or ">" in sample_str:
            return "ORDINAL", "category", "Grouped range (Ordinal)", ""
        return "NUMERIC_COUNT", "int64", "Discrete count value (Integer)", ""

    # 5. DATETIME
    if any(kw in col_lower for kw in ['ngày', 'tháng', 'year', 'date']):
        # Check if it's "1 THÁNG", "2 THÁNG"
        if "tháng" in col_lower and ("tháng " in sample_str or any(x in sample_str for x in ['1', '2', '3', '4', '5'])):
             return "CATEGORICAL", "category", "Interval description (e.g., '1 THÁNG') - recurring status", ""
        
        # Check for Excel serial
        risk = "Excel serial date detected" if any(len(str(x)) == 5 and str(x).isdigit() for x in [sample_val]) else ""
        return "DATETIME", "datetime64[ns]", "Time-series data", risk

    # 6. ORDINAL
    if any(kw in col_lower for kw in ['nhóm', 'level', 'grade', 'pos', 'vl theo mã']):
        return "ORDINAL", "category", "Ranked or grouped category (Ordinal)", ""

    # 7. BOOLEAN
    if unique_count <= 2 and any(v in sample_str for v in ['0', '1', 'có', 'không', 'true', 'false']):
        return "BOOLEAN", "bool", "Binary flag", ""

    # 8. CATEGORICAL (Nominal)
    # Low cardinality or specific keywords
    if unique_count < 100 or any(kw in col_lower for kw in ['phân loại', 'dự án', 'sản phẩm', 'chi nhánh', 'tỉnh', 'phụ trách', 'lead', 'giới tính', 'pl', 'quận', 'phường']):
        return "CATEGORICAL", "category", "Nominal category name", ""

    # 9. TEXT
    # If not caught by above, it's likely a description or name
    # Treat names, addresses as TEXT (string for AI)
    if any(kw in col_lower for kw in ['name', 'tên', 'địa chỉ', 'nơi', 'ghi chú', 'note', 'tên công ty', 'đdpl', 'tham chiếu', 'quan hệ', 'tình trạng liên hệ']):
        return "TEXT", "string", "Free text or long descriptive string", ""

    return "TEXT", "string", "Undetermined high-cardinality string", "Review manually"

# Apply the refined logic
results = []
for idx, row in df.iterrows():
    tag, dtype, reason, risk = get_refined_ai_schema(
        row['Tên Cột'], 
        row['Ví Dụ'], 
        row['Kiểu Dữ Liệu Suy Luận (Financial)'], 
        row['Lý Do'],
        row['Số Lượng Phân Biệt (Sample)'],
        row['Giá Trị Phống (Null)']
    )
    results.append({
        'Tên Cột': row['Tên Cột'],
        'Nhóm (Tag)': tag,
        'Pandas Dtype': dtype,
        'Mô Tả / Lý Do': reason,
        'Risky Column?': risk,
        'Sample': row['Ví Dụ']
    })

refined_df = pd.DataFrame(results)
refined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"Refined AI schema saved to {output_csv}")

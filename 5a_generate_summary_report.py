"""
Financial Data Cleaning - Summary Report Generator
====================================================
Tạo báo cáo tóm tắt chi tiết về quá trình xử lý dữ liệu
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ==================== CONFIG ====================
ORIGINAL_FILE = "TỔNG HỢP NĂM 2026.csv"
CLEANED_FILE = "TỔNG HỢP NĂM 2026 CLEANED.csv"
SCHEMA_FILE = "refined_schema_ai.csv"
OUTPUT_REPORT = "SUMMARY_REPORT.md"

# ==================== LOAD DATA ====================
print(" Loading original and cleaned datasets...")
try:
    # Use low_memory=False to avoid DtypeWarning, and only load as strings if necessary
    # Original might be messy, let's load first few lines to check
    df_original = pd.read_csv(ORIGINAL_FILE, encoding='utf-8', low_memory=False)
    df_cleaned = pd.read_csv(CLEANED_FILE, encoding='utf-8', low_memory=False)
    schema_df = pd.read_csv(SCHEMA_FILE, encoding='utf-8')
except FileNotFoundError as e:
    print(f" Error: {e}")
    # Try alternate paths if needed or exit
    import os
    if not os.path.exists(ORIGINAL_FILE):
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
    exit(1)

print(f"✓ Original: {df_original.shape[0]:,} rows x {df_original.shape[1]:,} columns")
print(f"✓ Cleaned:  {df_cleaned.shape[0]:,} rows x {df_cleaned.shape[1]:,} columns")


# ==================== BUILD SCHEMA DICT ====================
schema = {}
for idx, row in schema_df.iterrows():
    col_name = row['Tên Cột']
    schema[col_name] = {
        'tag': row['Nhóm (Tag)'],
        'dtype': row['Pandas Dtype'],
        'is_risky': bool(row['Risky Column?']),
        'description': row['Mô Tả / Lý Do']
    }

# ==================== ANALYZE ====================
report = []

def section(title):
    report.append(f"\n{'='*80}\n# {title}\n{'='*80}\n")

def subsection(title):
    report.append(f"\n## {title}\n")

def table_row(cols):
    report.append("| " + " | ".join(str(c) for c in cols) + " |")

# ==================== REPORT ====================
report.append("#  FINANCIAL DATA CLEANING & ANALYTICAL SUMMARY REPORT\n")
report.append(f"**Generated**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
report.append(f"**Dataset**: Distressed Debt Portfolio (Bad Debt Management)\n")
report.append(f"**Source File**: `{ORIGINAL_FILE}`\n")
report.append(f"**Cleaned File**: `{CLEANED_FILE}`\n")

# 1. OVERVIEW
section("1. DATASET EXECUTIVE SUMMARY")
report.append("Báo cáo này cung cấp cái nhìn tổng quan về chất lượng dữ liệu sau quá trình làm sạch và chuẩn hóa. " 
              "Dữ liệu tập trung vào quản lý nợ xấu, bao gồm thông tin khách hàng, số dư nợ, lịch sử thanh toán và trạng thái hồ sơ.\n\n")

table_row(["Metric", "Value"])
table_row(["-" * 25, "-" * 30])
table_row(["Total Record Count", f"{df_original.shape[0]:,}"])
table_row(["Total Original Columns", f"{df_original.shape[1]:,}"])
table_row(["Total Cleaned Columns", f"{df_cleaned.shape[1]:,}"])
table_row(["Cleaning Status", " COMPLETED"])


# 2. DATA TYPE CASTING
section("2. DATA TYPE CASTING RESULTS")
subsection("Summary")

tag_counts = {}
for col in df_cleaned.columns:
    if col in schema:
        tag = schema[col]['tag']
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

table_row(["Data Type Tag", "Count", "% of Columns"])
table_row(["-" * 20, "-" * 10, "-" * 20])
for tag in sorted(tag_counts.keys()):
    pct = (tag_counts[tag] / len(df_cleaned.columns)) * 100
    table_row([tag, tag_counts[tag], f"{pct:.1f}%"])

subsection("Columns by Type")
for tag in sorted(tag_counts.keys()):
    cols_in_tag = [c for c in df_cleaned.columns if c in schema and schema[c]['tag'] == tag]
    if len(cols_in_tag) <= 20:
        report.append(f"\n### {tag} ({len(cols_in_tag)} columns)\n")
        for col in cols_in_tag[:20]:
            report.append(f"- `{col}`\n")
    else:
        report.append(f"\n### {tag} ({len(cols_in_tag)} columns)\n")
        report.append(f"[First 20 of {len(cols_in_tag)}]\n")
        for col in cols_in_tag[:20]:
            report.append(f"- `{col}`\n")

# 3. MISSING VALUES ANALYSIS
section("3. MISSING VALUES ANALYSIS")
subsection("Summary")

missing_summary = {}
for col in df_cleaned.columns:
    n_missing = df_cleaned[col].isna().sum()
    missing_summary[col] = n_missing

total_missing_cells = sum(missing_summary.values())
missing_pct = (total_missing_cells / (df_cleaned.shape[0] * df_cleaned.shape[1])) * 100

table_row(["Metric", "Value"])
table_row(["-" * 30, "-" * 30])
table_row(["Total Cells", f"{df_cleaned.shape[0] * df_cleaned.shape[1]:,}"])
table_row(["Missing Cells", f"{total_missing_cells:,}"])
table_row(["Missing %", f"{missing_pct:.2f}%"])
table_row(["Columns with Missing", sum(1 for v in missing_summary.values() if v > 0)])

subsection("Columns with Most Missing Values")
sorted_missing = sorted(missing_summary.items(), key=lambda x: x[1], reverse=True)
table_row(["Column", "Missing Count", "Missing %"])
table_row(["-" * 30, "-" * 15, "-" * 12])
for col, n_missing in sorted_missing[:20]:
    if n_missing > 0:
        missing_pct_col = (n_missing / df_cleaned.shape[0]) * 100
        tag = schema.get(col, {}).get('tag', 'UNKNOWN')
        report.append(f"| {col[:35]:35s} | {n_missing:>8,d} | {missing_pct_col:>8.2f}% | {tag} |\n")

# 4. CRITICAL FINANCIAL COLUMNS
section("4. CRITICAL FINANCIAL COLUMNS")
subsection("Amounts (Monetary Values)")

amount_cols = [c for c in df_cleaned.columns if c in schema and schema[c]['tag'] == 'NUMERIC_AMOUNT']
report.append(f"Total: {len(amount_cols)} columns\n\n")
report.append("| Column | Count | Mean | Min | Max | Missing |\n")
report.append("|--------|-------|------|-----|-----|----------|\n")

for col in amount_cols[:15]:
    valid_data = pd.to_numeric(df_cleaned[col], errors='coerce').dropna()
    if len(valid_data) > 0:
        n_missing = df_cleaned[col].isna().sum()
        report.append(f"| {col[:40]:40s} | {len(valid_data):>6,d} | {valid_data.mean():>12,.0f} | {valid_data.min():>12,.0f} | {valid_data.max():>12,.0f} | {n_missing:>6,d} |\n")

subsection("Date Columns")
date_cols = [c for c in df_cleaned.columns if c in schema and schema[c]['tag'] == 'DATETIME']
# Include also any column that we manually added to datetime list in cleaner
if 'NGÀY THÁNG NĂM SINH' in df_cleaned.columns and 'NGÀY THÁNG NĂM SINH' not in date_cols:
    date_cols.append('NGÀY THÁNG NĂM SINH')

report.append(f"Total: {len(date_cols)} columns\n\n")

for col in date_cols[:15]:
    valid_dates_raw = df_cleaned[col].dropna()
    n_missing = df_cleaned[col].isna().sum()
    if len(valid_dates_raw) > 0:
        try:
            # Explicitly parse DD/MM/YYYY
            valid_dates = pd.to_datetime(valid_dates_raw, format='%d/%m/%Y', errors='coerce').dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                report.append(f"- **{col}**\n")
                report.append(f"  - Range: {min_date.strftime('%d/%m/%Y')} to {max_date.strftime('%d/%m/%Y')}\n")
                report.append(f"  - Valid: {len(valid_dates):,} | Missing: {n_missing:,}\n")
        except Exception as e:
            pass


# 5. DATA QUALITY ISSUES
section("5. DATA QUALITY ISSUES & ANOMALIES")
subsection("Negative Values in Amount Columns")

for col in amount_cols:
    try:
        numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce')
        neg_count = (numeric_col < 0).sum()
        if neg_count > 0:
            report.append(f"- **{col}**: {neg_count:,} negative values\n")
    except:
        pass

subsection("Outliers (> 5σ in Amount Columns)")

for col in amount_cols:
    try:
        numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce').dropna()
        if len(numeric_col) > 10:
            mean = numeric_col.mean()
            std = numeric_col.std()
            outliers = (numeric_col > mean + 5*std).sum()
            if outliers > 0:
                report.append(f"- **{col}**: {outliers:,} potential outliers (mean={mean:,.0f}, σ={std:,.0f})\n")
    except:
        pass

# 6. RISKY COLUMNS
section("6. RISKY COLUMNS (FLAGGED FOR MANUAL REVIEW)")

risky_cols = [c for c in df_cleaned.columns if c in schema and schema[c]['is_risky']]
report.append(f"Total: {len(risky_cols)} risky columns\n\n")

risky_by_type = {}
for col in risky_cols:
    tag = schema[col]['tag']
    if tag not in risky_by_type:
        risky_by_type[tag] = []
    risky_by_type[tag].append(col)

for tag in sorted(risky_by_type.keys()):
    report.append(f"### {tag}\n")
    for col in risky_by_type[tag]:
        report.append(f"- {col}\n")

# 7. RECOMMENDATIONS
section("7. RECOMMENDATIONS & NEXT STEPS")

recommendations = [
    "**DateTime Columns**: Nhiều cột datetime chứa giá trị không parse được (G.0903, 01-500.26, etc). Cần investigate dữ liệu gốc để xác định format đúng.",
    "**Amount Column Handling**: Các cột NUMERIC_AMOUNT không được fill missing values - điều này là đúng theo domain tài chính. Cần phân tích tại sao missing để có quyết định business logic hợp lý.",
    "**TUỔI (Age)**: Có 1,372 missing values không được fill vì context không rõ. Có thể suy luận từ NGÀY THÁNG NĂM SINH nếu cần.",
    "**Outliers**: Phát hiện nhiều outliers > 5σ trong các amount columns. Cần validate liệu đây có phải legitimate extreme values hay data entry errors.",
    "**Negative Values**: Phát hiện giá trị âm trong SỐ TIỀN TT GẦN NHẤT - cần kiểm tra logic nghiệp vụ.",
    "**Category Columns**: Nhiều category columns được fill bằng mode, có thể cân nhắc tạo category 'Unknown' hoặc 'Missing' thay vì dùng mode.",
    "**Data Validation**: Nên thực hiện cross-validation logic (vd: payment_date < due_date, amount > 0, etc) trước khi đưa vào model."
]

for i, rec in enumerate(recommendations, 1):
    report.append(f"{i}. {rec}\n\n")

# 8. FILES GENERATED
section("8. OUTPUT FILES")

report.append(f"- **FILE_TỔNG_03.26_CLEANED.csv** ({df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]:,} cols) - Main cleaned dataset\n")
report.append(f"- **data_cleaning_report.txt** - Detailed full log with all processing steps\n")
report.append(f"- **SUMMARY_REPORT.md** - This summary report (Markdown format)\n")

# ==================== SAVE REPORT ====================
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"\n Report saved to: {OUTPUT_REPORT}")
print(f" Total lines: {len(report)}")
es: {len(report)}")

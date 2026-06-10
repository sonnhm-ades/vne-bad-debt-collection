import pandas as pd
import numpy as np

# Use the file found in list_dir
csv_path = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026.csv'
output_path = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\schema_inference_output.csv'

def infer_financial_type(col_name, series):
    """
    Carefully infer the type based on Vietnamese financial domain.
    """
    col_lower = col_name.lower()
    
    # Placeholder/Common nulls
    placeholders = ['n/a', 'unknown', '-', 'none', 'null', 'nan']
    # Filter out placeholders for better inference
    clean_series = series.astype(str).str.lower().replace(placeholders, np.nan).dropna()
    
    if clean_series.empty:
        return 'unknown', "All values are placeholders or nulls"

    # 1. Identifiers (Mã, ID, Hợp đồng, KH) - Do not encode/scale
    id_keywords = ['mã', 'id', 'hợp đồng', 'cid', 'contract', 'mã kh', 'account', 'số phiếu']
    if any(kw in col_lower for kw in id_keywords):
        return 'identifier', f"Matched identifier keyword: '{col_name}'"

    # 2. Datetime (Ngày)
    date_keywords = ['ngày', 'date', 'tháng', 'year']
    if any(kw in col_lower for kw in date_keywords):
        # Sample conversion check
        try:
            pd.to_datetime(clean_series.head(5), errors='raise')
            return 'datetime', "Pattern matches date format and keyword 'ngày'"
        except:
            pass

    # 3. Numeric (Sô dư, Tiền, Lãi, Nợ, %...)
    amount_keywords = ['tiền', 'số dư', 'lãi', 'nợ', 'gốc', 'amount', 'balance', 'principal', 'debt', 'pht', 'phí', 'fee']
    # Convert to numeric to check
    numeric_series = pd.to_numeric(clean_series, errors='coerce')
    if numeric_series.notnull().sum() > len(clean_series) * 0.8: # Most values are numeric
        if any(kw in col_lower for kw in amount_keywords) or numeric_series.nunique() > 20:
             return 'numeric', f"Numeric values detected with keyword/cardinality support"
        else:
             return 'categorical', f"Numeric values but low uniqueness ({numeric_series.nunique()}), likely a code/state"

    # 4. Boolean (Flag, 0/1, Có/Không)
    unique_vals = clean_series.unique()
    if len(unique_vals) <= 2:
        val_set = set(str(v).lower() for v in unique_vals)
        bool_positives = {'1', '1.0', 'true', 'có', 'yes', 'y', 't'}
        bool_negatives = {'0', '0.0', 'false', 'không', 'no', 'n', 'f'}
        if val_set.issubset(bool_positives.union(bool_negatives)):
            return 'boolean', f"Binary values detected: {list(unique_vals)}"

    # 5. Categorical (Everything else that matches string patterns)
    if clean_series.nunique() < len(clean_series) * 0.05 or clean_series.nunique() < 100:
        return 'categorical', f"Categorical values with {clean_series.nunique()} unique types"

    return 'unknown/description', "High cardinality string without clear keyword"

def run_analysis():
    print(f"Reading {csv_path} (loading 20,000 rows for analysis)...")
    try:
        # Load a substantial sample for accurate inference
        df = pd.read_csv(csv_path, nrows=20000, low_memory=False)
        print(f"Read sample of {len(df)} rows.")
        
        analysis_data = []
        for col in df.columns:
            series = df[col]
            dtype = series.dtype
            
            # Count logical markers
            null_count = series.isna().sum()
            placeholders = ['n/a', 'N/A', 'unknown', '-', 'NONE', 'None', 'NULL']
            placeholder_count = series.astype(str).isin(placeholders).sum()
            
            # Infer Type
            inferred_type, reason = infer_financial_type(col, series)
            
            # Outliers (Numeric only)
            outliers_desc = "N/A"
            if inferred_type == 'numeric':
                n_series = pd.to_numeric(series, errors='coerce').dropna()
                if not n_series.empty:
                    q1 = n_series.quantile(0.25)
                    q3 = n_series.quantile(0.75)
                    iqr = q3 - q1
                    outliers_count = n_series[(n_series < (q1 - 1.5 * iqr)) | (n_series > (q3 + 1.5 * iqr))].count()
                    outliers_desc = f"{outliers_count} potential outliers (IQR)"

            analysis_data.append({
                'Tên Cột': col,
                'Kiểu Dữ Liệu Thực Tế (Pandas)': dtype,
                'Kiểu Dữ Liệu Suy Luận (Financial)': inferred_type,
                'Lý Do': reason,
                'Giá Trị Phống (Null)': null_count,
                'Giá Trị Placeholder (N/A, -)': placeholder_count,
                'Số Lượng Phân Biệt (Sample)': series.nunique(),
                'Ngoại Lệ (Outliers)': outliers_desc,
                'Ví Dụ': series.dropna().unique()[:3].tolist()
            })

        results_df = pd.DataFrame(analysis_data)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"--- SUCCESS ---")
        print(f"Schema results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_analysis()

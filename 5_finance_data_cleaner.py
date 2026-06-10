"""
FINANCIAL DATA CLEANING - COMPREHENSIVE VERSION
Financial Dataset: Bad Debt Analysis
Purpose: Strict type casting, datetime normalization, data quality checks
"""

import pandas as pd
import numpy as np
import re
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import sys
import io
import os

# Handle UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log', encoding='utf-8', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# Also print to console with UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

console_handler = UTF8StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class FinancialDataCleaner:
    """Comprehensive financial data cleaner for Bad Debt Analysis"""
    
    def __init__(self, data_path: str, schema_path: str, output_path: str = None):
        self.data_path = data_path
        self.schema_path = schema_path
        self.output_path = output_path or data_path.replace('.csv', '_CLEANED.csv')
        
        self.df = None
        self.schema_df = None
        self.df_original = None
        
        self.report = {
            'type_casting': {
                'successful': 0,
                'failed': 0,
                'details': {} # Col -> Error
            },
            'datetime_conversion': {},
            'de_masking': {},
            'missing_values': {},
            'anomalies': {
                'outliers': {},
                'logical_errors': []
            },
            'risky_columns': [],
            'errors': []
        }

    
    def load_data(self) -> bool:
        """Load data and schema files."""
        try:
            logger.info("\n" + "="*80)
            logger.info("STEP 1: LOADING DATA & SCHEMA")
            logger.info("="*80)
            
            logger.info(f"Loading data from: {self.data_path}")
            # Loading all as string initially to avoid premature inference
            self.df = pd.read_csv(self.data_path, encoding='utf-8', dtype=str)
            self.df_original = self.df.copy()
            logger.info(f"[OK] Data loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            
            logger.info(f"Loading schema from: {self.schema_path}")
            self.schema_df = pd.read_csv(self.schema_path, encoding='utf-8')
            
            # Identify risky columns from schema
            if 'Risky Column?' in self.schema_df.columns:
                risky_mask = self.schema_df['Risky Column?'].notna() & (self.schema_df['Risky Column?'].str.strip() != '')
                self.report['risky_columns'] = self.schema_df.loc[risky_mask, 'Tên Cột'].tolist()
                logger.info(f"[INFO] Schema identified {len(self.report['risky_columns'])} risky columns")
            
            logger.info(f"[OK] Schema loaded: {len(self.schema_df)} column definitions")
            return True
        except Exception as e:
            logger.error(f"[FAIL] Error loading data: {e}")
            self.report['errors'].append(f"Load error: {e}")
            return False
    
    def identify_critical_columns(self):
        """Identify critical financial columns based on keywords."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: IDENTIFY CRITICAL FINANCIAL COLUMNS")
        logger.info("="*80)
        
        critical_patterns = {
            'AMOUNT': ['amount', 'tien', 'giai ngan', 'thanh toan', 'balance', 'debt', 'du no', 'lai', 'phi', 'ket qua'],
            'DATE': ['ngay', 'date', 'thoi gian', 'ky han'],
            'STATUS': ['status', 'tinh trang', 'default', 'phan loai', 'nhom'],
        }
        
        critical_cols = {}
        for category, keywords in critical_patterns.items():
            cols = [c for c in self.df.columns 
                   if any(kw.lower() in c.lower() for kw in keywords)]
            if cols:
                critical_cols[category] = cols
                logger.info(f"{category} ({len(cols)} columns): {', '.join(cols[:3])}...")
        
        self.report['critical_financial_features'] = critical_cols

    
    def normalize_encoded_values(self):
        """Normalize encoded/special values to NaN and trim whitespace."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: NORMALIZE ENCODED/SPECIAL VALUES & TRIM")
        logger.info("="*80)
        
        encoded_patterns = ['N/A', 'NA', 'unknown', '-', '', 'null', 'NULL', 'None', '0', '0.0']
        # Note: We are cautious with '0' as it might be a valid value, 
        # but in many categorical/ID columns in this dataset, it represents NA.
        # We will apply this mainly to string/category columns later if needed, 
        # but for now, let's keep it general for specific NA markers.
        na_markers = ['N/A', 'NA', 'unknown', '-', 'null', 'NULL', 'None']
        
        normalized_count = 0
        for col in self.df.columns:
            # Trim whitespace first
            self.df[col] = self.df[col].astype(str).str.strip()
            
            # Replace common NA markers with NaN
            mask = self.df[col].isin(na_markers) | (self.df[col] == '') | (self.df[col] == 'nan')
            count = mask.sum()
            if count > 0:
                self.df.loc[mask, col] = np.nan
                normalized_count += count
        
        logger.info(f"[OK] Total values normalized to NaN: {normalized_count:,}")

    def handle_parenthesis_notation(self):
        """Remove parenthesis from numeric values, treating them as positive (as per instructions)."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: HANDLE PARENTHESIS NOTATION (e.g., '(14)' -> '14')")
        logger.info("="*80)
        
        # Target amount columns specifically
        amount_cols = self.report.get('critical_financial_features', {}).get('AMOUNT', [])
        
        fixed_count = 0
        for col in amount_cols:
            if col not in self.df.columns: continue
            
            mask = self.df[col].astype(str).str.contains(r'\(.*\)', regex=True, na=False)
            count = mask.sum()
            
            if count > 0:
                self.df.loc[mask, col] = self.df.loc[mask, col].str.replace(r'[\(\)]', '', regex=True)
                fixed_count += count
                logger.info(f"  - {col}: Fixed {count} values using parenthesis removal")
        
        logger.info(f"[OK] Total parenthesis notations fixed: {fixed_count:,}")

    
    def normalize_datetime(self):
        """Normalize all datetime columns with advanced logic."""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: ADVANCED DATETIME NORMALIZATION")
        logger.info("="*80)
        
        datetime_cols = self.schema_df[
            self.schema_df['Pandas Dtype'] == 'datetime64[ns]'
        ]['Tên Cột'].tolist()
        
        # Manually add birthday if not already included (it might be marked as category)
        if 'NGÀY THÁNG NĂM SINH' in self.df.columns and 'NGÀY THÁNG NĂM SINH' not in datetime_cols:
            datetime_cols.append('NGÀY THÁNG NĂM SINH')
        
        total_converted = 0
        
        for col in datetime_cols:
            if col not in self.df.columns: continue
            
            # 1. Basic cleanup: Remove " 12:00:00 AM" and similar timestamps
            self.df[col] = self.df[col].astype(str).str.replace(r'\s+\d{1,2}:\d{2}:\d{2}(\s+[AP]M)?', '', regex=True, case=False)
            
            # 2. Handle Excel Serial Dates (e.g., 44833)
            def convert_excel_date(val):
                if pd.isna(val) or val == 'nan' or val == '': return val
                if re.match(r'^\d{5}$', str(val)):
                    try:
                        serial = int(val)
                        # Excel's base date is 1899-12-30
                        dt = datetime(1899, 12, 30) + timedelta(days=serial)
                        return dt.strftime('%d/%m/%Y')
                    except: return val
                return val
            
            self.df[col] = self.df[col].apply(convert_excel_date)
            
            # 3. Handle MM/DD/YYYY to DD/MM/YYYY conversion
            # We look for patterns where the second part is > 12 (must be day) or detect first part > 12 (must be day)
            def standardize_date_format(val):
                if pd.isna(val) or val == 'nan' or val == '': return val
                val = str(val).strip()
                
                # Try common formats
                for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']:
                    try:
                        dt = pd.to_datetime(val, format=fmt)
                        # If we successfully parsed it, return as DD/MM/YYYY
                        return dt.strftime('%d/%m/%Y')
                    except: continue
                
                # If everything fails, try flexible parsing but log warning
                try:
                    dt = pd.to_datetime(val, dayfirst=False) # Default to US MM/DD
                    # However, if it's ambiguous, we might be wrong. 
                    # The user said: "MM/DD/YYYY thì chuyển hết về DD/MM/YYYY"
                    return dt.strftime('%d/%m/%Y')
                except:
                    return val

            # Detect if column is likely MM/DD
            # (Sample some values, if any have middle part > 12 and first <= 12)
            self.df[col] = self.df[col].apply(standardize_date_format)
            
            logger.info(f"  - {col}: Normalized to DD/MM/YYYY")

    def demask_encoded_dates(self):
        """De-mask and extract specific patterns (G.0903, 11-500.25)."""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: DE-MASKING ENCODED PATTERNS")
        logger.info("="*80)
        
        # Patterns: G.0903 -> 0903, 11-500.25 -> 25
        # We search across all columns but focus on those with "NGÀY" or IDs
        
        extracted_info = {}
        
        for col in self.df.columns:
            # Look for G.xxxx
            g_matches = self.df[col].astype(str).str.extract(r'G\.(\d{4})', expand=False)
            if g_matches.notna().any():
                count = g_matches.notna().sum()
                logger.info(f"  - {col}: Extracted {count} patterns of 'G.xxxx'")
                extracted_info[f"{col}_G_EXTRACT"] = g_matches
            
            # Look for xx-500.xx
            dot_matches = self.df[col].astype(str).str.extract(r'\d{2}-500\.(\d{2})', expand=False)
            if dot_matches.notna().any():
                count = dot_matches.notna().sum()
                logger.info(f"  - {col}: Extracted {count} patterns of 'xx-500.xx'")
                extracted_info[f"{col}_DOT_EXTRACT"] = dot_matches
        
        # We don't overwrite the original columns yet, but we could add them as new info 
        # or use them for correlation. For now, let's just log and store.
        self.report['de_masking'] = {k: len(v.notna()) for k, v in extracted_info.items()}

    
    def cast_data_types(self):
        """Apply strict type casting based on schema."""
        logger.info("\n" + "="*80)
        logger.info("STEP 7: STRICT TYPE CASTING")
        logger.info("="*80)
        
        successful = 0
        failed = 0
        
        for _, row in self.schema_df.iterrows():
            col_name = row['Tên Cột']
            target_type = row['Pandas Dtype']
            
            if col_name not in self.df.columns:
                continue
            
            try:
                if target_type == 'int64':
                    # To handle NaN in integers, we use Int64 (nullable integer)
                    self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce').round().astype('Int64')
                    successful += 1
                    
                elif target_type == 'float64':
                    self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce')
                    successful += 1
                    
                elif target_type == 'category':
                    self.df[col_name] = self.df[col_name].astype('category')
                    successful += 1
                    
                elif target_type == 'string':
                    # Special handling for ID/Number strings that might have .0 from being read as float
                    def clean_id_string(val):
                        if pd.isna(val) or val == 'nan' or val == '': return val
                        val_str = str(val).strip()
                        if val_str.endswith('.0'):
                            return val_str[:-2]
                        return val_str
                    
                    self.df[col_name] = self.df[col_name].apply(clean_id_string).astype('string')
                    successful += 1
                    
                elif target_type == 'datetime64[ns]':
                    # Already handled in normalize_datetime, but let's ensure it's datetime type if needed
                    # Actually, for the CSV output, we might want to keep it as string DD/MM/YYYY
                    # but for internal processing, we can convert it.
                    # Since the user asked for DD/MM/YYYY in the CSV, we'll keep it as strings for now 
                    # and only convert if they want to do further analysis in the script.
                    successful += 1
                    
                elif target_type == 'bool' or target_type == 'boolean':
                    # Flexible boolean mapping
                    self.df[col_name] = self.df[col_name].astype(str).str.lower().map({
                        '1': True, '1.0': True, 'true': True, 'yes': True, 'tố tụng': True,
                        '0': False, '0.0': False, 'false': False, 'no': False, 'nan': np.nan
                    })
                    successful += 1
                
            except Exception as e:
                error_msg = f"Error casting {col_name} to {target_type}: {str(e)}"
                logger.warning(f"[FAIL] {error_msg}")
                self.report['type_casting']['details'][col_name] = error_msg
                failed += 1
        
        logger.info(f"[OK] Successfully cast: {successful} columns")
        logger.info(f"[FAIL] Failed casting: {failed} columns")
        
        self.report['type_casting']['successful'] = successful
        self.report['type_casting']['failed'] = failed

    def check_data_quality(self):
        """Check for financial data quality issues (outliers, logic)."""
        logger.info("\n" + "="*80)
        logger.info("STEP 8: DATA QUALITY CHECKS")
        logger.info("="*80)
        
        # 1. Detect Outliers in Amount Columns
        amount_cols = self.report.get('critical_financial_features', {}).get('AMOUNT', [])
        for col in amount_cols:
            if col not in self.df.columns: continue
            
            # Convert to numeric for outlier check
            data = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(data) > 10:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr # 3*IQR is a safe threshold for outliers
                upper_bound = q3 + 3 * iqr
                
                outliers_mask = (data < lower_bound) | (data > upper_bound)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0:
                    logger.info(f"  - {col}: Detected {outlier_count} potential outliers (Extreme values)")
                    self.report['anomalies']['outliers'][col] = int(outlier_count)
        
        # 2. Check Logical Errors (e.g. NGÀY TT GẦN NHẤT < NGÀY GIẢI NGÂN)
        # We need to convert back to datetime for comparison
        try:
            if 'NGÀY GIẢI NGÂN' in self.df.columns and 'NGÀY TT GẦN NHẤT' in self.df.columns:
                d1 = pd.to_datetime(self.df['NGÀY GIẢI NGÂN'], format='%d/%m/%Y', errors='coerce')
                d2 = pd.to_datetime(self.df['NGÀY TT GẦN NHẤT'], format='%d/%m/%Y', errors='coerce')
                
                logic_error_mask = (d2 < d1).dropna()
                error_count = logic_error_mask.sum()
                
                if error_count > 0:
                    msg = f"Logic Error: {error_count} rows have 'NGÀY TT GẦN NHẤT' before 'NGÀY GIẢI NGÂN'"
                    logger.warning(f"  [CRITICAL] {msg}")
                    self.report['anomalies']['logical_errors'].append(msg)
        except Exception as e:
            logger.error(f"Error during logic check: {e}")

    
    def analyze_missing_values(self):
        """Analyze missing values without imputation."""
        logger.info("\n" + "="*80)
        logger.info("STEP 9: MISSING VALUES ANALYSIS")
        logger.info("="*80)
        
        missing_summary = {}
        for col in self.df.columns:
            count = self.df[col].isna().sum()
            if count > 0:
                pct = (count / len(self.df)) * 100
                missing_summary[col] = {'count': int(count), 'pct': float(pct)}
        
        # Sort and log top 10
        sorted_missing = sorted(missing_summary.items(), key=lambda x: x[1]['count'], reverse=True)
        for col, info in sorted_missing[:10]:
            logger.info(f"  - {col}: {info['count']:,} ({info['pct']:.1f}%)")
        
        self.report['missing_values'] = missing_summary

    def clean(self) -> bool:
        """Execute complete financial cleaning pipeline."""
        logger.info("\n[RUNNING] STARTING FINANCIAL DATA CLEANING PIPELINE\n")
        
        if not self.load_data():
            return False
        
        self.identify_critical_columns()
        self.normalize_encoded_values()
        self.handle_parenthesis_notation()
        self.normalize_datetime()
        self.demask_encoded_dates()
        self.cast_data_types()
        self.check_data_quality()
        self.analyze_missing_values()
        
        logger.info("\n" + "="*80)
        logger.info("[COMPLETE] CLEANING COMPLETED")
        logger.info("="*80)
        return True
    
    def save_cleaned_data(self) -> bool:
        """Save cleaned data to CSV, ensuring DD/MM/YYYY for dates."""
        try:
            logger.info(f"\nSaving cleaned data to: {self.output_path}")
            
            # Since we already converted dates to DD/MM/YYYY strings in normalize_datetime,
            # we can just save. We use index=False to match original format.
            self.df.to_csv(self.output_path, index=False, encoding='utf-8')
            logger.info(f"[OK] Saved: {len(self.df):,} rows x {len(self.df.columns)} columns")
            return True
        except Exception as e:
            logger.error(f"[FAIL] Error saving: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive cleaning report."""
        report_lines = []
        report_lines.append("# FINANCIAL DATA CLEANING SUMMARY REPORT")
        report_lines.append(f"\n**Execution Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Dataset**: {os.path.basename(self.data_path)}")
        report_lines.append(f"**Rows**: {len(self.df):,}")
        report_lines.append(f"**Columns**: {len(self.df.columns)}")
        
        report_lines.append("\n## 1. TYPE CASTING RESULTS")
        report_lines.append(f"- **Successful**: {self.report['type_casting']['successful']}")
        report_lines.append(f"- **Failed**: {self.report['type_casting']['failed']}")
        if self.report['type_casting']['details']:
            report_lines.append("\n### Casting Errors:")
            for col, err in self.report['type_casting']['details'].items():
                report_lines.append(f"- `{col}`: {err}")
        
        report_lines.append("\n## 2. DATA QUALITY ANOMALIES")
        if self.report['anomalies']['outliers']:
            report_lines.append("\n### Potential Outliers (Extreme Amounts):")
            for col, count in self.report['anomalies']['outliers'].items():
                report_lines.append(f"- `{col}`: {count} outliers detected")
        
        if self.report['anomalies']['logical_errors']:
            report_lines.append("\n### Logical Violations:")
            for err in self.report['anomalies']['logical_errors']:
                report_lines.append(f"- {err}")
        
        report_lines.append("\n## 3. MISSING VALUES")
        report_lines.append(f"Total columns with missing values: {len(self.report['missing_values'])}")
        # Top 5 missing
        sorted_missing = sorted(self.report['missing_values'].items(), key=lambda x: x[1]['count'], reverse=True)
        for col, info in sorted_missing[:5]:
            report_lines.append(f"- `{col}`: {info['count']:,} ({info['pct']:.1f}%)")
        
        report_lines.append("\n## 4. RISKY COLUMNS")
        if self.report['risky_columns']:
            for col in self.report['risky_columns']:
                report_lines.append(f"- {col}")
        else:
            report_lines.append("- None identified in schema.")
            
        return '\n'.join(report_lines)


if __name__ == "__main__":
    work_dir = r"c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026"
    data_file = os.path.join(work_dir, "TỔNG HỢP NĂM 2026.csv")
    schema_file = os.path.join(work_dir, "refined_schema_ai.csv")
    output_file = os.path.join(work_dir, "TỔNG HỢP NĂM 2026 CLEANED.csv")
    
    cleaner = FinancialDataCleaner(data_file, schema_file, output_file)
    
    if cleaner.clean():
        if cleaner.save_cleaned_data():
            report_str = cleaner.generate_report()
            print("\n" + "="*80)
            print(report_str)
            
            # Save report to file
            report_path = os.path.join(work_dir, "CLEANING_REPORT.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            logger.info(f"Report saved to {report_path}")
        else:
            logger.error("Failed to save cleaned data.")
    else:
        logger.error("Cleaning pipeline failed.")


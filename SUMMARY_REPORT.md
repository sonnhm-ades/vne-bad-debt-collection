#  FINANCIAL DATA CLEANING & ANALYTICAL SUMMARY REPORT

**Generated**: 15/04/2026 10:47:04

**Dataset**: Distressed Debt Portfolio (Bad Debt Management)

**Source File**: `TỔNG HỢP NĂM 2026.csv`

**Cleaned File**: `TỔNG HỢP NĂM 2026 CLEANED.csv`


================================================================================
# 1. DATASET EXECUTIVE SUMMARY
================================================================================

Báo cáo này cung cấp cái nhìn tổng quan về chất lượng dữ liệu sau quá trình làm sạch và chuẩn hóa. Dữ liệu tập trung vào quản lý nợ xấu, bao gồm thông tin khách hàng, số dư nợ, lịch sử thanh toán và trạng thái hồ sơ.


| Metric | Value |
| ------------------------- | ------------------------------ |
| Total Record Count | 244,695 |
| Total Original Columns | 76 |
| Total Cleaned Columns | 76 |
| Cleaning Status |  COMPLETED |

================================================================================
# 2. DATA TYPE CASTING RESULTS
================================================================================


## Summary

| Data Type Tag | Count | % of Columns |
| -------------------- | ---------- | -------------------- |
| BOOLEAN | 1 | 1.3% |
| CATEGORICAL | 29 | 38.2% |
| DATETIME | 4 | 5.3% |
| EMPTY | 2 | 2.6% |
| ID | 9 | 11.8% |
| NUMERIC_AMOUNT | 13 | 17.1% |
| NUMERIC_COUNT | 4 | 5.3% |
| NUMERIC_RATIO | 1 | 1.3% |
| ORDINAL | 4 | 5.3% |
| TEXT | 9 | 11.8% |

## Columns by Type


### BOOLEAN (1 columns)

- `HỒ SƠ KHỞI KIỆN`


### CATEGORICAL (29 columns)

[First 20 of 29]

- `GIỚI TÍNH`

- `NGÀY THÁNG NĂM SINH`

- `PHƯỜNG TẠM TRÚ`

- `QUẬN/HUYỆN TẠM TRÚ`

- `TỈNH TẠM TRÚ`

- `PHƯỜNG THƯỜNG TRÚ`

- `QUẬN/HUYỆN THƯỜNG TRÚ`

- `TỈNH THƯỜNG TRÚ`

- `CHỨC VỤ`

- `PL NGHÀNH NGHỀ`

- `SẢN PHẨM`

- `VNE LAW PL 01`

- `VNE LAW PL 02`

- `PHÂN LOẠI VÙNG MIỀN`

- `ĐÁNH GIÁ KHÁCH HÀNG`

- `DỰ ÁN`

- `ĐỐI TÁC PL1`

- `ĐỐI TÁC PL2`

- `SỐ CUỘC GỌI T01`

- `SỐ PHÚT GỌI T01`


### DATETIME (4 columns)

- `NGÀY GIẢI NGÂN`

- `NGÀY KẾT THÚC HĐ`

- `NGÀY TT GẦN NHẤT`

- `NGÀY LÀM VIỆC`


### EMPTY (2 columns)

- `PHÍ PHẠT`

- `NGÀY DỰ KIẾN CÓ KẾT QUẢ`


### ID (9 columns)

- `LOAN ID`

- `HỢP ĐỒNG SỐ 2`

- `SỐ LƯỢNG HỢP ĐỒNG`

- `CCCD/CMND`

- `MÃ TÌNH TRẠNG LIÊN HỆ`

- `MÃ BHXH KHÁCH HÀNG`

- `TÌNH TRẠNG MÃ BHYT`

- `PHÂN LOẠI VL THEO MÃ`

- `MÃ ĐKKCBBD`


### NUMERIC_AMOUNT (13 columns)

- `MỨC LƯƠNG`

- `SỐ TIỀN GIẢI NGÂN`

- `PHÍ BẢO HIỂM`

- `SỐ TIỀN THANH TOÁN HÀNG THÁNG`

- `SỐ TIỀN TT GẦN NHẤT`

- `NỢ GỐC`

- `LÃI TRONG HẠN`

- `LÃI QUÁ HẠN`

- `TỔNG NỢ`

- `SỐ TIỀN THÔNG BÁO`

- `DỰ KIẾN KẾT QUẢ`

- `KẾT QUẢ`

- `NGÀY CÓ KẾT QUẢ `


### NUMERIC_COUNT (4 columns)

- `TUỔI`

- `SỐ KỲ`

- `SỐ KỲ ĐÃ TT`

- `DPD`


### NUMERIC_RATIO (1 columns)

- `LÃI SUẤT`


### ORDINAL (4 columns)

- `PHÂN LOẠI POS`

- `PL NHÓM TUỔI`

- `NHÓM DPD`

- `NHÓM SỐ LẦN TT`


### TEXT (9 columns)

- `ĐỊA CHỈ TẠM TRÚ`

- `ĐỊA CHỈ THƯỜNG TRÚ`

- `SỐ ĐIỆN THOẠI.1`

- `TỔNG ĐÃ THANH TOÁN`

- `MỤC TIÊU ĐỐI TÁC`

- `TÌNH TRẠNG LIÊN HỆ`

- `NƠI ĐKKCBBD`

- `ĐỊA CHỈ`

- `SỐ HD DÀI HANMIR`


================================================================================
# 3. MISSING VALUES ANALYSIS
================================================================================


## Summary

| Metric | Value |
| ------------------------------ | ------------------------------ |
| Total Cells | 18,596,820 |
| Missing Cells | 5,201,855 |
| Missing % | 27.97% |
| Columns with Missing | 68 |

## Columns with Most Missing Values

| Column | Missing Count | Missing % |
| ------------------------------ | --------------- | ------------ |
| SỐ CUỘC GỌI T01                     |  244,604 |    99.96% | CATEGORICAL |

| SỐ PHÚT GỌI T01                     |  244,604 |    99.96% | CATEGORICAL |

| MỤC TIÊU VNE T01                    |  244,572 |    99.95% | CATEGORICAL |

| NGÀY CÓ KẾT QUẢ                     |  241,382 |    98.65% | NUMERIC_AMOUNT |

| KẾT QUẢ                             |  241,354 |    98.63% | NUMERIC_AMOUNT |

| THÁNG ĐỀ XUẤT KHỞI KIỆN             |  241,057 |    98.51% | CATEGORICAL |

| HỒ SƠ KHỞI KIỆN                     |  239,726 |    97.97% | BOOLEAN |

| ĐỊA CHỈ                             |  237,046 |    96.87% | TEXT |

| HỢP ĐỒNG SỐ 2                       |  217,464 |    88.87% | ID |

| NGÀY DỰ KIẾN CÓ KẾT QUẢ             |  214,265 |    87.56% | EMPTY |

| DỰ KIẾN KẾT QUẢ                     |  213,641 |    87.31% | NUMERIC_AMOUNT |

| PHÍ BẢO HIỂM                        |  191,964 |    78.45% | NUMERIC_AMOUNT |

| MỨC LƯƠNG                           |  177,794 |    72.66% | NUMERIC_AMOUNT |

| CHỨC VỤ                             |  173,067 |    70.73% | CATEGORICAL |

| TÌNH TRẠNG SMS                      |  167,859 |    68.60% | CATEGORICAL |

| TÌNH TRẠNG LIÊN HỆ                  |  165,306 |    67.56% | TEXT |

| MÃ TÌNH TRẠNG LIÊN HỆ               |  165,279 |    67.54% | ID |

| NGÀY LÀM VIỆC                       |  165,270 |    67.54% | DATETIME |

| PHÍ PHẠT                            |  156,046 |    63.77% | EMPTY |

| SỐ ĐIỆN THOẠI.1                     |  147,857 |    60.43% | TEXT |


================================================================================
# 4. CRITICAL FINANCIAL COLUMNS
================================================================================


## Amounts (Monetary Values)

Total: 13 columns


| Column | Count | Mean | Min | Max | Missing |

|--------|-------|------|-----|-----|----------|

| MỨC LƯƠNG                                | 66,901 |   12,869,850 |            0 |  350,000,000 | 177,794 |

| SỐ TIỀN GIẢI NGÂN                        | 238,490 |   34,273,995 |            0 | 7,500,000,000 |  6,205 |

| PHÍ BẢO HIỂM                             | 52,731 |    1,710,030 |            0 |    6,975,000 | 191,964 |

| SỐ TIỀN THANH TOÁN HÀNG THÁNG            | 221,966 |    3,013,466 |            0 |  423,420,378 | 22,729 |

| SỐ TIỀN TT GẦN NHẤT                      | 212,108 |    1,651,484 |  -73,850,000 |  320,000,000 | 32,587 |

| NỢ GỐC                                   | 244,684 |   24,742,947 |            0 | 30,000,000,000 |     11 |

| LÃI TRONG HẠN                            | 212,117 |   14,774,667 |  -14,764,295 | 3,454,712,444 | 32,578 |

| LÃI QUÁ HẠN                              | 204,060 |   35,454,757 |            0 | 72,885,000,000 | 40,635 |

| TỔNG NỢ                                  | 244,245 |   67,759,537 |            0 | 72,885,000,000 |    450 |

| SỐ TIỀN THÔNG BÁO                        | 221,958 |   47,164,890 |            0 | 2,950,451,582 | 22,737 |

| DỰ KIẾN KẾT QUẢ                          | 31,054 |       15,498 |            0 |   27,000,000 | 213,641 |

| KẾT QUẢ                                  |  3,341 |    5,088,854 |        1,000 |  116,640,487 | 241,354 |

| NGÀY CÓ KẾT QUẢ                          |  3,313 |       46,069 |       46,023 |       46,112 | 241,382 |


## Date Columns

Total: 5 columns


- **NGÀY GIẢI NGÂN**

  - Range: 11/06/2007 to 06/02/2026

  - Valid: 226,290 | Missing: 715

- **NGÀY KẾT THÚC HĐ**

  - Range: 01/01/2012 to 03/06/2039

  - Valid: 197,392 | Missing: 4,584

- **NGÀY TT GẦN NHẤT**

  - Range: 26/05/2009 to 28/02/2026

  - Valid: 157,070 | Missing: 33,659

- **NGÀY LÀM VIỆC**

  - Range: 26/11/2025 to 26/03/2026

  - Valid: 79,354 | Missing: 165,270

- **NGÀY THÁNG NĂM SINH**

  - Range: 30/03/1948 to 19/12/2013

  - Valid: 244,191 | Missing: 504


================================================================================
# 5. DATA QUALITY ISSUES & ANOMALIES
================================================================================


## Negative Values in Amount Columns

- **SỐ TIỀN TT GẦN NHẤT**: 28 negative values

- **LÃI TRONG HẠN**: 186 negative values


## Outliers (> 5σ in Amount Columns)

- **MỨC LƯƠNG**: 156 potential outliers (mean=12,869,850, σ=9,649,573)

- **SỐ TIỀN GIẢI NGÂN**: 363 potential outliers (mean=34,273,995, σ=40,909,958)

- **PHÍ BẢO HIỂM**: 25 potential outliers (mean=1,710,030, σ=843,693)

- **SỐ TIỀN THANH TOÁN HÀNG THÁNG**: 2,153 potential outliers (mean=3,013,466, σ=6,797,443)

- **SỐ TIỀN TT GẦN NHẤT**: 789 potential outliers (mean=1,651,484, σ=2,788,344)

- **NỢ GỐC**: 55 potential outliers (mean=24,742,947, σ=108,468,887)

- **LÃI TRONG HẠN**: 873 potential outliers (mean=14,774,667, σ=28,212,143)

- **LÃI QUÁ HẠN**: 27 potential outliers (mean=35,454,757, σ=286,954,584)

- **TỔNG NỢ**: 57 potential outliers (mean=67,759,537, σ=287,980,032)

- **SỐ TIỀN THÔNG BÁO**: 485 potential outliers (mean=47,164,890, σ=41,435,714)

- **DỰ KIẾN KẾT QUẢ**: 96 potential outliers (mean=15,498, σ=321,186)

- **KẾT QUẢ**: 30 potential outliers (mean=5,088,854, σ=10,608,832)


================================================================================
# 6. RISKY COLUMNS (FLAGGED FOR MANUAL REVIEW)
================================================================================

Total: 76 risky columns


### BOOLEAN

- HỒ SƠ KHỞI KIỆN

### CATEGORICAL

- GIỚI TÍNH

- NGÀY THÁNG NĂM SINH

- PHƯỜNG TẠM TRÚ

- QUẬN/HUYỆN TẠM TRÚ

- TỈNH TẠM TRÚ

- PHƯỜNG THƯỜNG TRÚ

- QUẬN/HUYỆN THƯỜNG TRÚ

- TỈNH THƯỜNG TRÚ

- CHỨC VỤ

- PL NGHÀNH NGHỀ

- SẢN PHẨM

- VNE LAW PL 01

- VNE LAW PL 02

- PHÂN LOẠI VÙNG MIỀN

- ĐÁNH GIÁ KHÁCH HÀNG

- DỰ ÁN

- ĐỐI TÁC PL1

- ĐỐI TÁC PL2

- SỐ CUỘC GỌI T01

- SỐ PHÚT GỌI T01

- MỤC TIÊU VNE T01

- TÌNH TRẠNG SMS

- TÌNH TRẠNG VL

- PHỤ TRÁCH HỒ SƠ

- LEAD QUẢN LÝ HỒ SƠ

- CHI NHÁNH

- THÁNG ĐỀ XUẤT KHỞI KIỆN

- KHÁCH HÀNG NHIỀU DỰ ÁN

- THÁNG

### DATETIME

- NGÀY GIẢI NGÂN

- NGÀY KẾT THÚC HĐ

- NGÀY TT GẦN NHẤT

- NGÀY LÀM VIỆC

### EMPTY

- PHÍ PHẠT

- NGÀY DỰ KIẾN CÓ KẾT QUẢ

### ID

- LOAN ID

- HỢP ĐỒNG SỐ 2

- SỐ LƯỢNG HỢP ĐỒNG

- CCCD/CMND

- MÃ TÌNH TRẠNG LIÊN HỆ

- MÃ BHXH KHÁCH HÀNG

- TÌNH TRẠNG MÃ BHYT

- PHÂN LOẠI VL THEO MÃ

- MÃ ĐKKCBBD

### NUMERIC_AMOUNT

- MỨC LƯƠNG

- SỐ TIỀN GIẢI NGÂN

- PHÍ BẢO HIỂM

- SỐ TIỀN THANH TOÁN HÀNG THÁNG

- SỐ TIỀN TT GẦN NHẤT

- NỢ GỐC

- LÃI TRONG HẠN

- LÃI QUÁ HẠN

- TỔNG NỢ

- SỐ TIỀN THÔNG BÁO

- DỰ KIẾN KẾT QUẢ

- KẾT QUẢ

- NGÀY CÓ KẾT QUẢ 

### NUMERIC_COUNT

- TUỔI

- SỐ KỲ

- SỐ KỲ ĐÃ TT

- DPD

### NUMERIC_RATIO

- LÃI SUẤT

### ORDINAL

- PHÂN LOẠI POS

- PL NHÓM TUỔI

- NHÓM DPD

- NHÓM SỐ LẦN TT

### TEXT

- ĐỊA CHỈ TẠM TRÚ

- ĐỊA CHỈ THƯỜNG TRÚ

- SỐ ĐIỆN THOẠI.1

- TỔNG ĐÃ THANH TOÁN

- MỤC TIÊU ĐỐI TÁC

- TÌNH TRẠNG LIÊN HỆ

- NƠI ĐKKCBBD

- ĐỊA CHỈ

- SỐ HD DÀI HANMIR


================================================================================
# 7. RECOMMENDATIONS & NEXT STEPS
================================================================================

1. **DateTime Columns**: Nhiều cột datetime chứa giá trị không parse được (G.0903, 01-500.26, etc). Cần investigate dữ liệu gốc để xác định format đúng.


2. **Amount Column Handling**: Các cột NUMERIC_AMOUNT không được fill missing values - điều này là đúng theo domain tài chính. Cần phân tích tại sao missing để có quyết định business logic hợp lý.


3. **TUỔI (Age)**: Có 1,372 missing values không được fill vì context không rõ. Có thể suy luận từ NGÀY THÁNG NĂM SINH nếu cần.


4. **Outliers**: Phát hiện nhiều outliers > 5σ trong các amount columns. Cần validate liệu đây có phải legitimate extreme values hay data entry errors.


5. **Negative Values**: Phát hiện giá trị âm trong SỐ TIỀN TT GẦN NHẤT - cần kiểm tra logic nghiệp vụ.


6. **Category Columns**: Nhiều category columns được fill bằng mode, có thể cân nhắc tạo category 'Unknown' hoặc 'Missing' thay vì dùng mode.


7. **Data Validation**: Nên thực hiện cross-validation logic (vd: payment_date < due_date, amount > 0, etc) trước khi đưa vào model.



================================================================================
# 8. OUTPUT FILES
================================================================================

- **FILE_TỔNG_03.26_CLEANED.csv** (244,695 rows × 76 cols) - Main cleaned dataset

- **data_cleaning_report.txt** - Detailed full log with all processing steps

- **SUMMARY_REPORT.md** - This summary report (Markdown format)
rmat)

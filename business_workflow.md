# Tái cấu trúc Dự án Phân tích Thu hồi Nợ 2026

## Bối cảnh & Vấn đề

Dự án hiện tại là một pipeline phân tích dữ liệu nợ xấu năm 2026, bao gồm các bước từ làm sạch dữ liệu đến mô hình hoá rủi ro. Code hoạt động nhưng có cấu trúc **"flat" (phẳng)** — tất cả các file Python nằm cùng một thư mục với data thô, báo cáo và các file tạm. Đây là dấu hiệu của một dự án được phát triển theo kiểu "script-by-script" mà không có kiến trúc từ trước.

---

## Những vấn đề nhận diện được

### ❌ Cấu trúc thư mục hiện tại (TRƯỚC)
```
TONG HOP NAM 2026/
├── .venv/                          # Virtual env nằm chung với code và data!
├── 1_data_understanding.py
├── 2_convert_to_csv.py
├── 3_automated_schema_inference.py
├── ...
├── 8i_rating_migration.py          # 25 file .py nằm phẳng!
├── TỔNG HỢP NĂM 2026.xlsx          # Data thô ~185MB nằm chung
├── TỔNG HỢP NĂM 2026.csv           # ~226MB
├── TỔNG HỢP NĂM 2026 CLEANED.csv   # ~228MB
├── refined_schema_ai.csv
├── schema_inference_output.csv
├── config_dpd.json                 # Config nằm rời rạc
├── data_cleaning.log               # Log nằm chung
├── deep_scan_report.txt
├── scratch_test_output.txt         # File tạm không được dọn
├── unique_codes.txt
├── CLEANING_REPORT.md
├── DATA_DICTIONARY.md
├── SUMMARY_REPORT.md
├── dpd.xlsx
├── partners/                       # CSV theo từng đối tác
├── pics/                           # Ảnh rời image004.png, ...
└── reports/
    ├── STRATEGY_HUB.html
    ├── STRATEGY_HUB.html.backup    # Backup file bị để lại!
    ├── Data_Science/
    │   ├── Data/                   # 18 file CSV output
    │   ├── Models/                 # pkl models
    │   └── Reports/                # 12 html reports
    └── Deep_Dive_Dashboards/       # 9 dashboard HTML
```

### Vấn đề cụ thể:
1. **Không có phân tách src/data/output** — Source code, dữ liệu thô, kết quả output nằm lẫn lộn
2. **Hardcoded paths** — Mỗi file .py đều có `r'c:\\Users\\HP\\Nextcloud\\SƠN - PHÂN TÍCH\\...'` hardcode
3. **Không có `config.py` trung tâm** — `config_dpd.json` đơn độc, không đủ
4. **Không có `requirements.txt` hay `pyproject.toml`** — Không ai biết cần cài gì
5. **Không có `README.md`** — Không có tài liệu hướng dẫn
6. **File tạm bị bỏ lại** — `scratch_test_output.txt`, `.html.backup`
7. **Naming convention lộn xộn** — vừa có `TỔNG HỢP NĂM 2026 CLEANED.csv` (tiếng Việt có dấu) vừa `PARTNER_ABBANK.csv`, `deep_scan_report.txt` (tiếng Anh không dấu)
8. **`pics/` không mô tả** — `image004.png`, `image005.png` — không biết là ảnh gì
9. **`.venv` nằm trong project thay vì ngoài** — Đây là vấn đề minor nhưng cũng không chuẩn
10. **Không có `__init__.py` hay module nào** — Không thể import lẫn nhau, không thể test

---

## Cấu trúc đề xuất (SAU)

```
debt-collection-2026/
│
├── README.md                       # [MỚI] Hướng dẫn tổng quan dự án
├── requirements.txt                # [MỚI] Dependencies Python
├── config.py                       # [MỚI] Cấu hình trung tâm (thay thế hardcoded paths)
├── .gitignore                      # [MỚI] Bỏ qua data lớn, .venv, __pycache__
│
├── data/
│   ├── raw/                        # Dữ liệu thô (KHÔNG sửa đổi)
│   │   ├── TONG_HOP_NAM_2026.xlsx
│   │   └── dpd.xlsx
│   ├── processed/                  # Dữ liệu sau làm sạch
│   │   └── TONG_HOP_NAM_2026_CLEANED.csv
│   ├── schema/                     # Metadata schema
│   │   ├── refined_schema_ai.csv
│   │   └── schema_inference_output.csv
│   └── partners/                   # Data phân tách theo đối tác
│       ├── PARTNER_ABBANK.csv
│       ├── PARTNER_SHB.csv
│       └── ...
│
├── src/                            # [MỚI] Toàn bộ source code
│   ├── __init__.py
│   │
│   ├── 01_ingestion/               # Giai đoạn 1: Nạp & hiểu dữ liệu
│   │   ├── __init__.py
│   │   ├── data_understanding.py   # (= 1_data_understanding.py)
│   │   └── convert_to_csv.py       # (= 2_convert_to_csv.py)
│   │
│   ├── 02_schema/                  # Giai đoạn 2: Suy luận schema
│   │   ├── __init__.py
│   │   ├── schema_inference.py     # (= 3_automated_schema_inference.py)
│   │   └── schema_refinement.py    # (= 4_refined_schema_analysis.py)
│   │
│   ├── 03_cleaning/                # Giai đoạn 3: Làm sạch dữ liệu
│   │   ├── __init__.py
│   │   └── finance_data_cleaner.py # (= 5_finance_data_cleaner.py)
│   │
│   ├── 04_partners/                # Giai đoạn 4: Phân tích theo đối tác
│   │   ├── __init__.py
│   │   └── partner_deep_dive.py    # (= 6_partner_deep_dive_pipeline.py)
│   │
│   ├── 05_data_science/            # Giai đoạn 5: Mô hình hoá
│   │   ├── __init__.py
│   │   ├── ptp_model.py            # (= 7a_data_science_ptp_model.py)
│   │   ├── segmentation.py         # (= 7b_data_science_segmentation.py)
│   │   ├── visualization.py        # (= 7c_insight_visualization.py)
│   │   ├── lgd_model.py            # (= 7d_risk_lgd_model.py)
│   │   ├── roll_rate_matrix.py     # (= 7e_risk_roll_rate_matrix.py)
│   │   └── survival_analysis.py    # (= 7f_risk_survival_analysis.py)
│   │
│   └── 06_deep_dive/               # Giai đoạn 6: Phân tích chuyên sâu
│       ├── __init__.py
│       ├── debt_aging.py           # (= 8a_debt_aging_analysis.py)
│       ├── contact_funnel.py       # (= 8b_contact_funnel_analysis.py)
│       ├── debt_stacking.py        # (= 8c_debt_stacking_analysis.py)
│       ├── geographic_analysis.py  # (= 8d_geographic_residual_analysis.py)
│       ├── product_vintage.py      # (= 8e_product_vintage_analysis.py)
│       ├── agent_performance.py    # (= 8f_agent_performance_attribution.py)
│       ├── master_dashboard.py     # (= 8g_master_dashboard.py)
│       ├── partner_risk.py         # (= 8h_partner_risk_decomposition.py)
│       └── rating_migration.py     # (= 8i_rating_migration.py)
│
├── reports/                        # Kết quả đầu ra (auto-generated)
│   ├── html/
│   │   ├── strategy_hub.html
│   │   ├── deep_dive/              # 9 dashboard partner
│   │   └── data_science/           # 12 html analysis reports
│   ├── data/                       # CSV outputs từ models
│   └── models/                     # Trained model artifacts (.pkl)
│
├── docs/                           # [MỚI] Tài liệu kỹ thuật
│   ├── DATA_DICTIONARY.md
│   ├── SUMMARY_REPORT.md
│   └── assets/                     # Ảnh dùng trong docs (thay thế pics/)
│       ├── arch_diagram.png
│       └── ...
│
└── logs/                           # [MỚI] Log files
    └── data_cleaning.log
```

---

## Những thay đổi code quan trọng

### 1. Tạo `config.py` trung tâm
Thay thế toàn bộ hardcoded path bằng:
```python
# config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA_FILE = RAW_DATA_DIR / "TONG_HOP_NAM_2026.xlsx"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "TONG_HOP_NAM_2026_CLEANED.csv"
```

### 2. Tạo `requirements.txt`
```
pandas>=2.0
numpy
scikit-learn
xgboost
shap
plotly
openpyxl
```

### 3. Tạo `README.md`
Hướng dẫn cài đặt, cách chạy pipeline, mô tả từng giai đoạn.

---

## Câu hỏi cần xác nhận

> [!IMPORTANT]
> **Phạm vi thực hiện**: Bạn muốn tôi làm gì?
> - **Option A** — Chỉ tái cấu trúc thư mục + tạo `config.py`, `requirements.txt`, `README.md`, `.gitignore` (không đụng vào code bên trong từng file)
> - **Option B** — Tái cấu trúc thư mục VÀ refactor code để dùng `config.py` chung, loại bỏ hardcoded paths trong tất cả các file
> - **Option C** — Làm tất cả ở trên + nâng cấp code quality (type hints, docstrings chuẩn, logging nhất quán)

> [!WARNING]
> **Dữ liệu lớn**: Các file `.xlsx` (~185MB) và `.csv` (~226MB, ~228MB) sẽ được DI CHUYỂN vào `data/raw/` và `data/processed/`. Điều này cần thời gian và dung lượng tạm (thao tác move chứ không copy). Bạn có đồng ý không?

> [!NOTE]
> **File tên tiếng Việt có dấu**: `TỔNG HỢP NĂM 2026.csv` → sẽ đổi thành `TONG_HOP_NAM_2026.csv` (không dấu, dùng underscore) vì một số công cụ CLI và môi trường Linux/server không xử lý tốt tên file Unicode. Bạn có đồng ý không?

---

## Kế hoạch thực hiện

1. **[Prep]** Tạo cấu trúc thư mục mới
2. **[Move]** Di chuyển files vào đúng vị trí (data, src, docs, logs)
3. **[Config]** Tạo `config.py`, `requirements.txt`, `README.md`, `.gitignore`
4. **[Refactor]** (Nếu Option B/C) Cập nhật import paths trong từng file Python
5. **[Clean]** Xóa các file tạm (`scratch_test_output.txt`, `.html.backup`, v.v.)
6. **[Verify]** Kiểm tra lại cấu trúc cuối cùng

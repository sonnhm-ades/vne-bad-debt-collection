# -*- coding: utf-8 -*-
"""
MODULE 8I — CUSTOMER RATING MIGRATION (Chuyển Dịch Xếp Hạng Khách Hàng)
Câu hỏi: ĐÁNH GIÁ KHÁCH HÀNG thay đổi A→D như thế nào qua 3 tháng?
         Hướng hạ hạng có phải Early Warning Indicator không?
Output:  reports/Data_Science/Reports/8i_RATING_MIGRATION.html + 8i_rating_migration.csv
"""
import pandas as pd
import numpy as np
import os, sys, warnings
from scipy.stats import chi2_contingency, pointbiserialr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science'
REPORT_DIR = os.path.join(OUTPUT_DIR, "Reports")
SUB_DATA_DIR = os.path.join(OUTPUT_DIR, "Data")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "Models")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SUB_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)




warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

FILE_PATH  = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'

MONTH_ORDER  = ['THÁNG 01.26', 'THÁNG 02.26', 'THÁNG 03.26']
RATING_ORDER = {'A': 4, 'B': 3, 'C': 2, 'D': 1, '0': 0}
RATING_LABELS = ['0', 'D', 'C', 'B', 'A']

def run():
    print("=" * 60)
    print("MODULE 8I — CUSTOMER RATING MIGRATION ANALYSIS")
    print("=" * 60)

    # ── 1. Load & pivot ────────────────────────────────────────
    print("\n[1/5] Nạp dữ liệu và pivot RATING theo tháng...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df['KẾT QUẢ'], errors='coerce').fillna(0)
    df['NỢ GỐC']  = pd.to_numeric(df['NỢ GỐC'],  errors='coerce').fillna(0)
    df['RATING']   = df['ĐÁNH GIÁ KHÁCH HÀNG'].astype(str).str.strip().str.upper()
    df['RATING']   = df['RATING'].replace({'NAN': '0', '': '0'})

    pivot = df.pivot_table(index='LOAN ID', columns='THÁNG',
                           values='ĐÁNH GIÁ KHÁCH HÀNG',
                           aggfunc='last').reset_index()
    pivot.columns.name = None

    # Chỉ lấy các cột tháng có trong data
    month_cols = [m for m in MONTH_ORDER if m in pivot.columns]
    print(f"   → {len(pivot):,} LOAN ID | Tháng có: {month_cols}")

    for m in month_cols:
        pivot[m] = pivot[m].astype(str).str.strip().str.upper().replace({'NAN': '0', '': '0'})
        pivot[f'{m}_NUM'] = pivot[m].map(RATING_ORDER).fillna(0)

    # ── 2. Tính chiều hướng migration ─────────────────────────
    print("[2/5] Tính chiều hướng Rating Migration ...")
    if len(month_cols) >= 2:
        first_m, last_m = month_cols[0], month_cols[-1]
        pivot['RATING_FIRST'] = pivot[first_m]
        pivot['RATING_LAST']  = pivot[last_m]
        pivot['RATING_FIRST_NUM'] = pivot[f'{first_m}_NUM']
        pivot['RATING_LAST_NUM']  = pivot[f'{last_m}_NUM']

        pivot['DELTA_RATING'] = pivot['RATING_LAST_NUM'] - pivot['RATING_FIRST_NUM']
        pivot['DIRECTION'] = pivot['DELTA_RATING'].apply(
            lambda d: 'NÂNG HẠNG 📈' if d > 0 else ('HẠ HẠNG 📉' if d < 0 else 'ỔN ĐỊNH ━')
        )
        
        # Thêm DPD để làm Early Warning Score
        dpd_pivot = df.pivot_table(index='LOAN ID', columns='THÁNG', values='DPD', aggfunc='last').reset_index()
        dpd_pivot.columns.name = None
        pivot['DPD_FIRST'] = dpd_pivot[first_m] if first_m in dpd_pivot.columns else 0
        pivot['DPD_LAST']  = dpd_pivot[last_m] if last_m in dpd_pivot.columns else 0
        pivot['EWS_RISK'] = np.where(
            (pivot['DELTA_RATING'] < 0) & (pivot['DPD_LAST'] > pivot['DPD_FIRST']),
            'HIGH RISK', 'NORMAL'
        )

    # Merge KẾT QUẢ (sum) và NỢ GỐC (last) per LOAN ID
    kq = df.groupby('LOAN ID').agg(
        KQ_TONG = ('KẾT QUẢ','sum'),
        NO_GOC  = ('NỢ GỐC','last'),
    ).reset_index()
    pivot = pivot.merge(kq, on='LOAN ID', how='left')
    pivot['CÓ_TRẢ'] = (pivot['KQ_TONG'] > 0).astype(int)

    # ── 3. Phân tích theo Direction ───────────────────────────
    print("[3/5] Phân tích Recovery theo Direction ...")
    dir_stats = pivot.groupby('DIRECTION').agg(
        SO_HS      = ('LOAN ID',     'count'),
        RATE_THU   = ('CÓ_TRẢ',     'mean'),
        DELTA_TB   = ('DELTA_RATING','mean'),
        NO_GOC_TB  = ('NO_GOC',      'mean'),
    ).reset_index()
    dir_stats['RATE_%'] = (dir_stats['RATE_THU'] * 100).round(2)
    print(dir_stats[['DIRECTION','SO_HS','RATE_%','DELTA_TB']].to_string(index=False))

    # Point-Biserial Correlation: DELTA_RATING vs CÓ_TRẢ
    valid = pivot.dropna(subset=['DELTA_RATING', 'CÓ_TRẢ'])
    if len(valid) > 30:
        corr, p_corr = pointbiserialr(valid['DELTA_RATING'], valid['CÓ_TRẢ'])
        print(f"\n   📊 Point-Biserial Correlation (ΔRATING ↔ CÓ_TRẢ):")
        print(f"      r={corr:.4f}, p={p_corr:.6f} — ",
              "Tương quan CÓ Ý NGHĨA" if p_corr < 0.05 else "Không đủ ý nghĩa")

    # ── 4. Migration Matrix (Tháng 1 → Tháng 3) ──────────────
    print("[4/5] Xây dựng Rating Migration Matrix ...")
    if len(month_cols) >= 2:
        mat_raw = pd.crosstab(pivot['RATING_FIRST'], pivot['RATING_LAST'])
        # Chỉ giữ hạng hợp lệ
        valid_ratings = [r for r in RATING_LABELS if r in mat_raw.index or r in mat_raw.columns]
        mat_raw = mat_raw.reindex(index=valid_ratings, columns=valid_ratings, fill_value=0)
        mat_pct = mat_raw.div(mat_raw.sum(axis=1), axis=0) * 100
        mat_pct = mat_pct.round(1)
        print("\n   Migration Matrix (% từ hàng→cột):")
        print(mat_pct.to_string())

        # Chi-square
        try:
            chi2, p_chi, dof, _ = chi2_contingency(mat_raw.values)
            chi_note = f"Chi-Square: χ²={chi2:.2f}, p={p_chi:.6f} → {'CÓ ý nghĩa thống kê' if p_chi < 0.05 else 'Không đủ ý nghĩa'}"
            print(f"\n   📊 {chi_note}")
        except Exception as e:
            chi_note = f"Chi-Square error: {e}"
    else:
        mat_pct, chi_note = pd.DataFrame(), "Không đủ 2 tháng để tính Migration Matrix"

    # ── 5. Visualization & Premium Layout ─────────────────────────────────────
    print("\n[5/5] Tạo Dashboard HTML cao cấp...")
    direction_order = ['HẠ HẠNG 📉', 'ỔN ĐỊNH ━', 'NÂNG HẠNG 📈']
    dir_stats_s = dir_stats.set_index('DIRECTION').reindex(direction_order).reset_index().dropna(subset=['SO_HS'])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Số Lượng Hồ Sơ Theo Hướng Dịch Chuyển",
            "Tỷ Lệ Thu Hồi Nợ Theo Hướng Dịch Chuyển (%)",
            "Ma Trận Dịch Chuyển Xếp Hạng (%) - Tháng Đầu → Cuối",
            "Cơ Cấu Xếp Hạng: Tháng Đầu vs Tháng Cuối",
        ),
        specs=[[{"type":"bar"},     {"type":"bar"}],
               [{"type":"heatmap"}, {"type":"bar"}]],
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )

    # Plot 1: Bar chart of counts by Direction
    fig.add_trace(go.Bar(
        x=dir_stats_s['DIRECTION'],
        y=dir_stats_s['SO_HS'],
        marker_color=['#EF4444', '#1E88E5', '#10B981'],
        name='Số hồ sơ',
        text=dir_stats_s['SO_HS'].apply(lambda v: f"{v:,}"),
        textposition='outside',
        showlegend=False
    ), row=1, col=1)

    # Plot 2: Paid rate by Direction
    fig.add_trace(go.Bar(
        x=dir_stats_s['DIRECTION'],
        y=dir_stats_s['RATE_%'],
        marker_color=['#EF4444', '#1E88E5', '#10B981'],
        name='Paid Rate (%)',
        text=dir_stats_s['RATE_%'].apply(lambda v: f"{v:.1f}%"),
        textposition='outside',
        showlegend=False
    ), row=1, col=2)

    # Plot 3: Heatmap of rating migration matrix
    if not mat_pct.empty:
        z_vals = mat_pct.values
        x_lbls = list(mat_pct.columns)
        y_lbls = list(mat_pct.index)
        fig.add_trace(go.Heatmap(
            z=z_vals, x=x_lbls, y=y_lbls,
            colorscale='YlOrRd',
            text=z_vals, texttemplate="%{text}%",
            colorbar=dict(title="Dịch Chuyển (%)", x=1.02),
            showscale=True,
        ), row=2, col=1)

    # Plot 4: Rating comparison (First vs Last)
    first_counts = pivot['RATING_FIRST'].value_counts().reindex(RATING_LABELS, fill_value=0)
    last_counts = pivot['RATING_LAST'].value_counts().reindex(RATING_LABELS, fill_value=0)
    fig.add_trace(go.Bar(
        name='Tháng Đầu', x=RATING_LABELS, y=first_counts.values,
        marker_color='#64748B', text=first_counts.values, textposition='outside'
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        name='Tháng Cuối', x=RATING_LABELS, y=last_counts.values,
        marker_color='#38BDF8', text=last_counts.values, textposition='outside'
    ), row=2, col=2)

    # Update fig layout to dark mode
    fig.update_layout(
        
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=850,
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=50, b=80, l=60, r=80),
    )
    # Remove figure annotations as we put them in HTML header
    fig.layout.annotations = []
    fig.update_xaxes(tickangle=-15, row=1, col=2)
    fig.update_xaxes(tickangle=-15, row=2, col=1)
    fig.update_yaxes(gridcolor='#334155')

    # Get HTML Plotly div
    import plotly.offline as plo
    chart_div = plo.plot(fig, output_type='div', include_plotlyjs=False)

    # Chart 5: Sankey Migration
    if len(month_cols) >= 2:
        nodes = [f"Bắt Đầu ({r})" for r in RATING_LABELS] + [f"Kết Thúc ({r})" for r in RATING_LABELS]
        node_indices = {name: i for i, name in enumerate(nodes)}
        sources, targets, values = [], [], []
        
        for r_first in RATING_LABELS:
            for r_last in RATING_LABELS:
                val = mat_raw.loc[r_first, r_last] if r_first in mat_raw.index and r_last in mat_raw.columns else 0
                if val > 0:
                    sources.append(node_indices[f"Bắt Đầu ({r_first})"])
                    targets.append(node_indices[f"Kết Thúc ({r_last})"])
                    values.append(val)
                    
        sankey_fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = nodes,
              color = '#38BDF8'
            ),
            link = dict(
              source = sources, target = targets, value = values,
              color = 'rgba(255, 255, 255, 0.1)'
          ))])
        sankey_fig.update_layout(title_text="Dòng Chuyển Dịch Xếp Hạng (Sankey Flow)", font=dict(family="Inter, sans-serif"),  height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
        sankey_div = plo.plot(sankey_fig, output_type='div', include_plotlyjs=False)
    else:
        sankey_div = "<p>Không đủ dữ liệu cho biểu đồ Sankey</p>"

    # Extract KPI metrics
    total_migrated = len(pivot)

    # Generate Data Tables HTML Rows
    dir_table_rows = ""
    for _, r in dir_stats.iterrows():
        rate_color = "#10B981" if r['RATE_%'] > 1 else "#EF4444"
        dir_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['DIRECTION']}</td>
            <td style="text-align:right;">{r['SO_HS']:,}</td>
            <td style="text-align:right; color:{rate_color}; font-weight:700;">{r['RATE_%']:.2f}%</td>
            <td style="text-align:right;">{r['DELTA_TB']:.2f}</td>
            <td style="text-align:right;">{r['NO_GOC_TB']/1e6:,.1f} Tr</td>
        </tr>"""

    mat_table_rows = ""
    if not mat_pct.empty:
        # Create headers dynamically based on mat_pct columns
        mat_headers = "".join([f"<th>Đến {col}</th>" for col in mat_pct.columns])
        for idx, row in mat_pct.iterrows():
            row_html = "".join([f'<td style="text-align:right;">{val:.1f}%</td>' for val in row])
            mat_table_rows += f"""
            <tr>
                <td style="font-weight:600; background:rgba(37,99,235,0.05);">Từ {idx}</td>
                {row_html}
            </tr>"""
    else:
        mat_headers = "<th>Không có dữ liệu</th>"
        mat_table_rows = "<tr><td>Trống</td></tr>"
    
    down_hs = len(pivot[pivot['DELTA_RATING'] < 0])
    down_pct = down_hs / total_migrated * 100 if total_migrated > 0 else 0.0
    
    up_hs = len(pivot[pivot['DELTA_RATING'] > 0])
    up_pct = up_hs / total_migrated * 100 if total_migrated > 0 else 0.0
    
    stable_hs = len(pivot[pivot['DELTA_RATING'] == 0])
    stable_pct = stable_hs / total_migrated * 100 if total_migrated > 0 else 0.0
    
    correlation_val = corr if 'corr' in locals() else 0.0
    chi_square_txt = chi_note if 'chi_note' in locals() else "Không tính toán"

    # Insights Generator Logic
    insights_html = ""
    high_risk_hs = len(pivot[pivot.get('EWS_RISK', '') == 'HIGH RISK'])
    if total_migrated > 0:
        high_risk_pct = high_risk_hs / total_migrated * 100
        if high_risk_pct > 10:
            insights_html += f"""
            <div class="alert-box alert-danger">
                <strong>🔴 CẢNH BÁO EWS (Early Warning Score):</strong> Có {high_risk_hs:,} hồ sơ ({high_risk_pct:.1f}%) vừa bị HẠ HẠNG tín dụng vừa có DPD TĂNG lên. Đây là nhóm có rủi ro vỡ nợ rất cao, cần chuyển ngay cho các Agent Elite xử lý ưu tiên thay vì theo dõi thông thường.
            </div>"""
        elif high_risk_hs > 0:
            insights_html += f"""
            <div class="alert-box alert-warn">
                <strong>🟡 Tín hiệu cảnh báo sớm:</strong> {high_risk_hs:,} hồ sơ có dấu hiệu xấu đi kép (hạ hạng + tăng DPD). Theo dõi chặt chẽ nhóm này.
            </div>"""
            
    if down_pct > 30:
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>⚠️ Danh mục suy giảm chất lượng:</strong> {down_pct:.1f}% khách hàng bị hạ hạng. Cần siết chặt các chính sách thu hồi và ngưng gia hạn nợ/tái cơ cấu cho các nhóm này vì khả năng phục hồi thấp.
        </div>"""
    elif up_pct > down_pct:
        insights_html += f"""
        <div class="alert-box alert-success">
            <strong>🟢 Chất lượng danh mục cải thiện:</strong> Tỷ lệ nâng hạng ({up_pct:.1f}%) đang cao hơn tỷ lệ hạ hạng ({down_pct:.1f}%).
        </div>"""
        
    if correlation_val > 0.15:
        insights_html += f"""
        <div class="alert-box alert-success">
            <strong>💡 Tính Hữu Dụng Của Xếp Hạng (Rating Utility):</strong> Có sự tương quan đồng biến (r={correlation_val:.2f}) rõ rệt giữa việc nâng/hạ hạng với kết quả trả nợ. Hệ thống xếp hạng nội bộ (Rating) đang hoạt động rất hiệu quả và dự báo chuẩn xác khả năng trả nợ. Khuyến nghị lấy Xếp hạng làm tiêu chí chính để phân luồng (Triage).
        </div>"""

    # Premium Layout HTML
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8I — Rating Migration Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg: #0F172A;
    --card: #1E293B;
    --primary: #38BDF8;
    --success: #10B981;
    --danger: #EF4444;
    --warn: #F59E0B;
    --muted: #64748B;
    --border: #334155;
    --text: #F1F5F9;
    --radius: 12px;
}}
body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    margin: 0;
}}
.header {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 24px;
}}
.header h1 {{
    margin: 0 0 8px 0;
    font-size: 20px;
    font-weight: 800;
    color: var(--primary);
    letter-spacing: -0.5px;
}}
.header p {{
    margin: 0;
    font-size: 13px;
    color: var(--muted);
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}
.kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    position: relative;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,.1);
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--primary);
    border-radius: 0 0 var(--radius) var(--radius);
}}
.kpi-card.kpi-success::after {{ background: var(--success); }}
.kpi-card.kpi-danger::after {{ background: var(--danger); }}
.kpi-card.kpi-warn::after {{ background: var(--warn); }}
.kpi-label {{
    font-size: 10px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}}
.kpi-value {{
    font-size: 24px;
    font-weight: 800;
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 6px;
}}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); margin-bottom: 0; }}
.tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; flex-wrap: wrap; }}
.data-table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.data-table th {{ background: #1E293B; padding: 10px 12px; font-weight: 700; color: var(--primary); border-bottom: 2px solid var(--border); text-align: left; white-space: nowrap; }}
.data-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
.data-table tr:hover {{ background: rgba(56,189,248,0.05); }}
.tab-btn {{ background: transparent; color: var(--muted); border: none; padding: 10px 16px; font-size: 14px; font-weight: 600; cursor: pointer; border-radius: 8px; transition: all 0.2s; }}
.tab-btn:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
.tab-btn.active {{ background: var(--primary); color: #000; }}
.tab-content {{ display: none; animation: fadeIn 0.3s ease; }}
.tab-content.active {{ display: block; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.alert-box {{ padding: 16px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; border-left: 4px solid; }}
.alert-danger {{ background: rgba(239, 68, 68, 0.1); border-left-color: var(--danger); color: #FCA595; }}
.alert-warn {{ background: rgba(245, 158, 11, 0.1); border-left-color: var(--warn); color: #FCD34D; }}
.alert-success {{ background: rgba(16, 185, 129, 0.1); border-left-color: var(--success); color: #6EE7B7; }}
        [data-theme="dark"] {{
            --background: #0F172A;
            --card-bg: #1E293B;
            --text-main: #F1F5F9;
            --text-muted: #94A3B8;
            --border: #334155;
            --primary-light: #38BDF8;
        }}
        [data-theme="dark"] .tab-headers {{ background: #1E293B; }}
        [data-theme="dark"] .tab-btn:hover {{ background: #2E3B4E; color: #38BDF8; }}
        [data-theme="dark"] .tab-btn.active {{ background: #0F172A; color: #38BDF8; border-bottom-color: #38BDF8; }}
        [data-theme="dark"] .data-table th {{ background: #1E293B; color: var(--text-main); }}
        [data-theme="dark"] .data-table tr:hover {{ background: #2E3B4E; }}
        [data-theme="dark"] .rec-card {{ background: #1E293B; border-color: var(--border); }}
        [data-theme="dark"] .rec-card h4 {{ color: #38BDF8; }}
        [data-theme="dark"] .rec-card p, [data-theme="dark"] .rec-card li {{ color: #CBD5E1; }}
        [data-theme="dark"] .callout.callout-info {{ background-color: rgba(59,130,246,0.15); color: #93C5FD; border-color: #3B82F6; }}
        [data-theme="dark"] .callout.callout-success {{ background-color: rgba(16,185,129,0.15); color: #6EE7B7; border-color: #10B981; }}
        [data-theme="dark"] .callout.callout-warning {{ background-color: rgba(245,158,11,0.15); color: #FCD34D; border-color: #F59E0B; }}
        [data-theme="dark"] .callout.callout-danger {{ background-color: rgba(239,68,68,0.15); color: #FCA5A5; border-color: #EF4444; }}


        /* ===================================================================
           VNE Light Mode — Injected for theme sync
           Activates when parent sends 'light' theme (data-theme is removed)
           =================================================================== */
        :root:not([data-theme="dark"]) {{
            /* Variable Overrides */
            --bg: #F8FAFC !important;
            --card: #FFFFFF !important;
            --text: #0F172A !important;
            --border: #E2E8F0 !important;
            --muted: #64748B !important;
            --primary: #2563EB !important;
            --accent: #2563EB !important;
            
            --background: #F8FAFC !important;
            --card-bg: #FFFFFF !important;
            --text-main: #0F172A !important;
            --text-muted: #64748B !important;
            
            --success: #10B981 !important;
            --danger: #EF4444 !important;
            --warning: #F59E0B !important;
            --warn: #F59E0B !important;
            --info: #06B6D4 !important;
        }}

        /* Force light mode backgrounds overriding native dark CSS */
        html:not([data-theme="dark"]) body {{
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }}

        html:not([data-theme="dark"]) .card,
        html:not([data-theme="dark"]) .kpi-card,
        html:not([data-theme="dark"]) .section-card,
        html:not([data-theme="dark"]) .chart-card,
        html:not([data-theme="dark"]) .metric-card,
        html:not([data-theme="dark"]) .insight-card,
        html:not([data-theme="dark"]) .stat-card,
        html:not([data-theme="dark"]) .summary-card,
        html:not([data-theme="dark"]) .detail-card,
        html:not([data-theme="dark"]) .analysis-card,
        html:not([data-theme="dark"]) .header,
        html:not([data-theme="dark"]) [class*="-card"],
        html:not([data-theme="dark"]) [class*="card-"] {{
            background-color: var(--card) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }}

        html:not([data-theme="dark"]) h1,
        html:not([data-theme="dark"]) h2,
        html:not([data-theme="dark"]) h3,
        html:not([data-theme="dark"]) h4,
        html:not([data-theme="dark"]) h5,
        html:not([data-theme="dark"]) h6 {{
            color: var(--text) !important;
        }}

        html:not([data-theme="dark"]) p,
        html:not([data-theme="dark"]) span:not([class*="badge"]):not([class*="alert"]):not([class*="kpi"]),
        html:not([data-theme="dark"]) label {{
            color: var(--text) !important;
        }}

        html:not([data-theme="dark"]) small, 
        html:not([data-theme="dark"]) .text-muted,
        html:not([data-theme="dark"]) .kpi-label,
        html:not([data-theme="dark"]) .kpi-sub {{ 
            color: var(--muted) !important; 
        }}

        /* Tables */
        html:not([data-theme="dark"]) table {{ border-color: var(--border) !important; }}
        html:not([data-theme="dark"]) th {{
            background-color: rgba(37,99,235,0.05) !important;
            color: #1E3A8A !important;
            border-color: var(--border) !important;
        }}
        html:not([data-theme="dark"]) td {{
            border-color: var(--border) !important;
            color: var(--text) !important;
        }}
        html:not([data-theme="dark"]) tr {{ border-color: var(--border) !important; }}
        html:not([data-theme="dark"]) tr:nth-child(even) {{ background-color: rgba(248,250,252,0.8) !important; }}
        html:not([data-theme="dark"]) tr:hover {{ background-color: rgba(37,99,235,0.05) !important; }}

        /* Section headers & dividers */
        html:not([data-theme="dark"]) .section-header,
        html:not([data-theme="dark"]) .header-section,
        html:not([data-theme="dark"]) .tabs,
        html:not([data-theme="dark"]) .divider,
        html:not([data-theme="dark"]) hr {{
            border-color: var(--border) !important;
        }}

        /* Tabs / navigation */
        html:not([data-theme="dark"]) .tab-btn,
        html:not([data-theme="dark"]) .nav-tab,
        html:not([data-theme="dark"]) .tab-link {{
            color: var(--muted) !important;
            background-color: transparent !important;
        }}
        html:not([data-theme="dark"]) .tab-btn:hover,
        html:not([data-theme="dark"]) .nav-tab:hover,
        html:not([data-theme="dark"]) .tab-link:hover {{
            color: var(--text) !important;
            background-color: rgba(0,0,0,0.05) !important;
        }}
        html:not([data-theme="dark"]) .tab-btn.active,
        html:not([data-theme="dark"]) .nav-tab.active,
        html:not([data-theme="dark"]) .tab-link.active {{
            color: var(--primary) !important;
            border-color: var(--primary) !important;
            background-color: rgba(37,99,235,0.08) !important;
        }}

        /* Input / form / footer */
        html:not([data-theme="dark"]) input,
        html:not([data-theme="dark"]) select,
        html:not([data-theme="dark"]) textarea,
        html:not([data-theme="dark"]) .footer {{
            background-color: var(--card) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }}
</style>
</head>
<body>
<div class="header">
    <h1>8I — CUSTOMER RATING MIGRATION</h1>
    <p>Phân tích biến động xếp hạng khách hàng: Đánh giá chiều hướng nâng/hạ hạng tín dụng nội bộ của khách hàng và tương quan với hiệu quả thu hồi nợ thực tế — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Số Khách Hàng Dịch Chuyển</div>
        <div class="kpi-value">{total_migrated:,} HS</div>
        <div class="kpi-sub">Số hồ sơ có lịch sử rating xuyên suốt</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Tỷ Lệ Hạ Hạng (Downgrade)</div>
        <div class="kpi-value">{down_pct:.2f}%</div>
        <div class="kpi-sub">{down_hs:,} hồ sơ bị giảm chất lượng</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Tỷ Lệ Nâng Hạng (Upgrade)</div>
        <div class="kpi-value">{up_pct:.2f}%</div>
        <div class="kpi-sub">{up_hs:,} hồ sơ tăng chất lượng</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Tương Quan Dịch Chuyển ↔ Thu Hồi</div>
        <div class="kpi-value">r = {correlation_val:.4f}</div>
        <div class="kpi-sub">Hạ hạng tín dụng tương quan nghịch với tỷ lệ trả nợ</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Tổng Quan Migration</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. Phân Tích Dòng Chảy Sankey</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. Ma Trận Dịch Chuyển 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Thống Kê Thay Đổi 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Insight & Khuyến Nghị</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="alert-box alert-warn" style="margin-bottom: 20px;">
        📢 <strong>Kết quả kiểm định Chi-Square:</strong> {chi_square_txt}
    </div>
    <div class="chart-card">{chart_div}</div>
</div>

<div id="tab2" class="tab-content">
    <div class="chart-card">{sankey_div}</div>
</div>

<div id="tab3" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📊 Ma Trận Dịch Chuyển Xếp Hạng Khách Hàng (%)</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Hạng Ban Đầu \\ Hạng Sau</th>
                {mat_headers}
            </tr></thead>
            <tbody>{mat_table_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab4" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📈 Thống Kê Theo Hướng Chuyển Dịch Tín Dụng</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Chiều Hướng (Direction)</th><th>Số Hồ Sơ</th><th>Tỷ Lệ Thu Hồi</th>
                <th>Mức Độ Dịch Chuyển (Trung Bình Hạng)</th><th>Nợ Gốc TB (VNĐ)</th>
            </tr></thead>
            <tbody>{dir_table_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Khuyến Nghị Cảnh Báo Sớm Dựa Trên Dữ Liệu</h3>
        {insights_html}
    </div>
</div>

<script>
function openTab(evt, tabName) {{
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].classList.remove("active");
    }}
    tablinks = document.getElementsByClassName("tab-btn");
    for (i = 0; i < tablinks.length; i++) {{
        tablinks[i].classList.remove("active");
    }}
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
    window.dispatchEvent(new Event('resize'));
}}

// Theme sync listener (f-string format)
(function() {{
    function updatePlotlyTheme(isDark) {{
        const textCol = isDark ? '#F1F5F9' : '#0F172A';
        const gridCol = isDark ? '#334155' : '#E2E8F0';
        const titleCol = isDark ? '#94A3B8' : '#64748B';
        
        document.querySelectorAll('.plotly-graph-div').forEach(div => {{
            try {{
                Plotly.relayout(div, {{
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'font.color': textCol,
                    'title.font.color': titleCol,
                    'xaxis.gridcolor': gridCol,
                    'yaxis.gridcolor': gridCol,
                    'xaxis.tickfont.color': textCol,
                    'yaxis.tickfont.color': textCol,
                    'xaxis.zerolinecolor': gridCol,
                    'yaxis.zerolinecolor': gridCol
                }});
            }} catch(e) {{}}
        }});
    }}

    const theme = localStorage.getItem('vne_theme');
    const initDark = (theme === 'dark');
    if (initDark) {{
        document.documentElement.setAttribute('data-theme', 'dark');
    }}
    
    // Listen for theme sync messages
    window.addEventListener('message', function(e) {{
        if (e.data && typeof e.data.theme === 'string') {{
            const isDark = (e.data.theme === 'dark');
            if (isDark) {{
                document.documentElement.setAttribute('data-theme', 'dark');
            }} else {{
                document.documentElement.removeAttribute('data-theme');
            }}
            setTimeout(() => {{
                updatePlotlyTheme(isDark);
                window.dispatchEvent(new Event('resize'));
            }}, 50);
        }}
    }});
    
    // Request initial theme
    try {{
        window.parent.postMessage({{ type: 'request_theme' }}, '*');
    }} catch(e) {{}}
    
    // Initial plotly theme adjustment on DOMContentLoaded
    window.addEventListener('DOMContentLoaded', () => {{
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        setTimeout(() => {{ updatePlotlyTheme(isDark); }}, 500);
    }});
}})();
</script>
</body>
</html>"""

    out_html = os.path.join(REPORT_DIR, "8i_RATING_MIGRATION.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    out_csv = os.path.join(SUB_DATA_DIR, "8i_rating_migration.csv")
    pivot[['LOAN ID','RATING_FIRST','RATING_LAST','DIRECTION','DELTA_RATING','CÓ_TRẢ','NO_GOC']].to_csv(
        out_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8I!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")

if __name__ == "__main__":
    run()
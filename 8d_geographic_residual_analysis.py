# -*- coding: utf-8 -*-
"""
MODULE 8D — PHÂN TÍCH ĐỊA LÝ DỰA TRÊN PHẦN DƯ + SKIP-TRACING (RESIDUAL GEOGRAPHIC + MIGRATION GAP)
Phiên bản: 2.0 (2026-05-26) — Tích hợp Migration Gap Score, CỜ_DI_CƯ, VÜNG MIỀN layer
Câu hỏi: Tỉnh/vùng nào thực sự khó đòi? Ai đang trốn nợ và đang ở đâu?
Biến nguồn (CLEANED.csv + DS_SEGMENTATION_FINAL.csv):
  - TỈNH TẠM TRÚ, TỈNH THƯỜNG TRÚ, KẾT QUẢ, DPD, NỢ GỐC
  - CỜ_DI_CƯ, TÌNH TRẠNG VL, PHÂN LOẠI VÜNG MIỀN (từ DS_SEG)
Migration Gap Score: CỜ_DI_CƯ=1 "AND" TẠM TRÚ≠THƯỜNG TRÚ "AND" VL NGỤY HIỂM → skip-trace cao
Output:  reports/Data_Science/GEO_RESIDUAL.html
"""
import pandas as pd
import numpy as np
import os, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
DS_SEG_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science\Data\DS_SEGMENTATION_FINAL.csv'

MIN_LOANS_PER_TINH = 200   # Bỏ qua tỉnh có quá ít hồ sơ để tránh nhiễu thống kê

def run():
    print("=" * 60)
    print("MODULE 8D — RESIDUAL GEOGRAPHIC ANALYSIS")
    print("=" * 60)

    # ── 1. Load & aggregate ────────────────────────────────────
    print("\n[1/5] Nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df.get('KẾT QUẢ'), errors='coerce').fillna(0)
    df['DPD']     = pd.to_numeric(df.get('DPD'),      errors='coerce')
    df['NỢ GỐC']  = pd.to_numeric(df.get('NỢ GỐC'),  errors='coerce').fillna(0)

    agg = df.groupby('LOAN ID').agg(
        KQ_TONG       = ('KẾT QUẢ',          'sum'),
        DPD_CUOI      = ('DPD',              'last'),
        NO_GOC        = ('NỢ GỐC',          'last'),
        TINH_TAM_TRU  = ('TỈNH TẠM TRÚ',    'first'),
        TINH_THUONG   = ('TỈNH THƯỜNG TRÚ', 'first'),
        VUNG_MIEN     = ('PHÂN LOẠI VÙNG MIỀN', 'first'),
    ).reset_index()
    agg = agg.rename(columns={'TINH_TAM_TRU': 'TINH'})

    # Merge DS_SEGMENTATION để lấy CỜ_DI_CƯ và TÌNH_TRẠNG_VL
    seg = pd.read_csv(DS_SEG_PATH, low_memory=False,
                      usecols=['LOAN ID', 'CỜ_DI_CƯ', 'TÌNH TRẠNG VL'])
    seg = seg.drop_duplicates(subset='LOAN ID', keep='last')
    agg = agg.merge(seg, on='LOAN ID', how='left')
    agg['CỜ_DI_CƯ'] = pd.to_numeric(agg['CỜ_DI_CƯ'], errors='coerce').fillna(0)

    # Migration Gap Score: 3 điều kiện lệch nhau → rủi ro skip-trace cao
    agg['DIA_LECH'] = (agg['TINH'] != agg['TINH_THUONG']).astype(int)
    VL_NGUY_HIEM = ['Thất nghiệp', 'Không có việc làm', 'Không rõ']
    agg['VL_RISK'] = agg['TÌNH TRẠNG VL'].isin(VL_NGUY_HIEM).astype(int)
    # MGS = số điều kiện lệch (0-3) → 3 = rủi ro skip-trace tối cao
    agg['MIGRATION_GAP_SCORE'] = agg['CỜ_DI_CƯ'] + agg['DIA_LECH'] + agg['VL_RISK']

    agg['TARGET'] = (agg['KQ_TONG'] > 0).astype(int)
    agg = agg.dropna(subset=['DPD_CUOI'])
    print(f"   → {len(agg):,} LOAN ID sau khi gom nhóm")
    skip_count = (agg['MIGRATION_GAP_SCORE'] == 3).sum()
    print(f"   → {skip_count:,} hồ sơ có MIGRATION_GAP_SCORE=3 (Rủi ro Skip-Trace Tối Cao)")

    # ── 2. Baseline Logistic Regression (loại trừ yếu tố gây nhiễu) ──
    print("[2/5] Xây dựng mô hình baseline (DPD + Nợ Gốc → Xác suất trả)...")
    feat_cols = ['DPD_CUOI', 'NO_GOC']
    X_raw = agg[feat_cols].fillna(0)
    y     = agg['TARGET']

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    lr.fit(X_sc, y)

    agg['BASELINE_PROB'] = lr.predict_proba(X_sc)[:, 1]
    # Residual = Thực tế - Dự báo baseline
    agg['RESIDUAL']      = agg['TARGET'] - agg['BASELINE_PROB']

    # ── 3. Gom residual theo Tỉnh ─────────────────────────────
    print("[3/5] Tổng hợp residual theo tỉnh/thành phố...")
    tinh_agg = agg.groupby('TINH').agg(
        SỐ_HỒ_SƠ         = ('LOAN ID',       'count'),
        RESIDUAL_TB       = ('RESIDUAL',      'mean'),
        TỶ_LỆ_THỰC_TẾ    = ('TARGET',        'mean'),
        TỶ_LỆ_DỰ_BÁO     = ('BASELINE_PROB', 'mean'),
    ).reset_index()

    tinh_agg = tinh_agg[tinh_agg['SỐ_HỒ_SƠ'] >= MIN_LOANS_PER_TINH].copy()
    tinh_agg['RESIDUAL_TB'] = tinh_agg['RESIDUAL_TB'] * 100   # → đổi sang %
    tinh_agg['TỶ_LỆ_THỰC_TẾ']  = (tinh_agg['TỶ_LỆ_THỰC_TẾ']  * 100).round(2)
    tinh_agg['TỶ_LỆ_DỰ_BÁO']   = (tinh_agg['TỶ_LỆ_DỰ_BÁO']   * 100).round(2)
    tinh_agg['RESIDUAL_TB']     = tinh_agg['RESIDUAL_TB'].round(3)

    tinh_agg = tinh_agg.sort_values('RESIDUAL_TB')
    print(f"   → {len(tinh_agg)} tỉnh đủ điều kiện phân tích (≥{MIN_LOANS_PER_TINH} HS)")
    print("\n Top 5 tỉnh KHÓ ĐÒI NHẤT (Residual âm lớn nhất):")
    print(tinh_agg.head(5)[['TINH','SỐ_HỒ_SƠ','TỶ_LỆ_THỰC_TẾ','TỶ_LỆ_DỰ_BÁO','RESIDUAL_TB']].to_string(index=False))
    print("\n Top 5 tỉnh DỄ THU HỒI NHẤT (Residual dương lớn nhất):")
    print(tinh_agg.tail(5)[['TINH','SỐ_HỒ_SƠ','TỶ_LỆ_THỰC_TẾ','TỶ_LỆ_DỰ_BÁO','RESIDUAL_TB']].to_string(index=False))

    # ── 4. Visualization ─────────────────────────────────────────
    print("\n[4/5] Tạo Dashboard HTML cao cấp...")
    import plotly.offline as plo

    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Residual Thu Hồi Theo Tỉnh (Thực Tế − Dự Báo)", "Phân Bổ Hồ Sơ Tỉnh", "Top 10 Khó Đòi Nhất", "Top 10 Dễ Thu Hồi Nhất"), specs=[[{"type": "bar", "colspan": 2}, None], [{"type": "bar"}, {"type": "bar"}]], vertical_spacing=0.15)
    
    bar_colors = ['#E53935' if v < 0 else '#43A047' for v in tinh_agg['RESIDUAL_TB']]
    fig1.add_trace(go.Bar(x=tinh_agg['TINH'], y=tinh_agg['RESIDUAL_TB'], marker_color=bar_colors, showlegend=False, hovertemplate="<b>%{x}</b><br>Residual: %{y:.3f}%<extra></extra>"), row=1, col=1)
    fig1.add_hline(y=0, line_dash="dash", line_color="#FDD835", row=1, col=1)
    
    worst = tinh_agg.head(10).sort_values('RESIDUAL_TB', ascending=True)
    fig1.add_trace(go.Bar(y=worst['TINH'], x=worst['RESIDUAL_TB'], orientation='h', marker_color='#E53935', text=[f"{v:.3f}%" for v in worst['RESIDUAL_TB']], textposition='outside', showlegend=False), row=2, col=1)
    
    best = tinh_agg.tail(10).sort_values('RESIDUAL_TB', ascending=True)
    fig1.add_trace(go.Bar(y=best['TINH'], x=best['RESIDUAL_TB'], orientation='h', marker_color='#43A047', text=[f"+{v:.3f}%" for v in best['RESIDUAL_TB']], textposition='outside', showlegend=False), row=2, col=2)
    
    fig1.update_layout( height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # ─── NEW: Vùng Miền Aggregation ──────────────────────────────
    if 'VUNG_MIEN' in agg.columns:
        vung_agg = agg.groupby('VUNG_MIEN').agg(
            SO_HS     = ('LOAN ID', 'count'),
            RATE      = ('TARGET', 'mean'),
            RESIDUAL  = ('RESIDUAL', 'mean'),
            SKIP_CT   = ('MIGRATION_GAP_SCORE', lambda x: (x == 3).sum()),
            EAD       = ('NO_GOC', 'sum'),
        ).reset_index().dropna(subset=['VUNG_MIEN'])
        vung_agg['RATE_%']      = (vung_agg['RATE'] * 100).round(2)
        vung_agg['RESIDUAL_%']  = (vung_agg['RESIDUAL'] * 100).round(3)
        vung_agg['EAD_TY']      = (vung_agg['EAD'] / 1e9).round(2)
        vung_agg['SKIP_PCT']    = (vung_agg['SKIP_CT'] / vung_agg['SO_HS'] * 100).round(1)
        vung_agg = vung_agg.sort_values('RESIDUAL_%')
    else:
        vung_agg = pd.DataFrame()

    # ─── NEW: Migration Gap Ranking by Province ────────────────
    mgs_by_tinh = agg.groupby('TINH').agg(
        SO_HS     = ('LOAN ID', 'count'),
        MGS3_CT   = ('MIGRATION_GAP_SCORE', lambda x: (x==3).sum()),
        MGS2_CT   = ('MIGRATION_GAP_SCORE', lambda x: (x==2).sum()),
        DI_CU     = ('CỜ_DI_CƯ', 'mean'),
        RESIDUAL  = ('RESIDUAL', 'mean'),
        EAD       = ('NO_GOC', 'sum'),
    ).reset_index()
    mgs_by_tinh['MGS3_%']     = (mgs_by_tinh['MGS3_CT'] / mgs_by_tinh['SO_HS'] * 100).round(1)
    mgs_by_tinh['DI_CU_%']    = (mgs_by_tinh['DI_CU'] * 100).round(1)
    mgs_by_tinh['RESIDUAL_%'] = (mgs_by_tinh['RESIDUAL'] * 100).round(3)
    mgs_by_tinh['EAD_TY']     = (mgs_by_tinh['EAD'] / 1e9).round(2)
    mgs_by_tinh = mgs_by_tinh[mgs_by_tinh['SO_HS'] >= MIN_LOANS_PER_TINH].sort_values('MGS3_%', ascending=False)

    # ─── NEW Fig 2: Regional Bar ──────────────────────────────────
    if not vung_agg.empty:
        vung_colors = ['#E53935' if v < 0 else '#43A047' for v in vung_agg['RESIDUAL_%']]
        fig2 = make_subplots(rows=1, cols=2,
            subplot_titles=("Residual Thu Hồi Theo Vùng Miền", "% Skip-Trace Cao (MGS=3) Theo Vùng"),
            horizontal_spacing=0.1)
        fig2.add_trace(go.Bar(
            x=vung_agg['VUNG_MIEN'], y=vung_agg['RESIDUAL_%'],
            marker_color=vung_colors, text=[f"{v:+.2f}%" for v in vung_agg['RESIDUAL_%']],
            textposition='outside', showlegend=False
        ), row=1, col=1)
        fig2.add_hline(y=0, line_dash="dash", line_color="#FDD835", row=1, col=1)
        fig2.add_trace(go.Bar(
            x=vung_agg['VUNG_MIEN'], y=vung_agg['SKIP_PCT'],
            marker_color='#F59E0B', text=[f"{v:.1f}%" for v in vung_agg['SKIP_PCT']],
            textposition='outside', showlegend=False
        ), row=1, col=2)
        fig2.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=50, b=50, l=40, r=40))
        div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)
    else:
        div2 = "<p style='color:var(--muted)'>Không có dữ liệu vùng miền.</p>"

    # ─── NEW Fig 3: MGS Ranking by Province ──────────────────────
    mgs_top = mgs_by_tinh.head(15)
    fig3 = make_subplots(rows=1, cols=2,
        subplot_titles=("Top 15 Tỉnh Có % MGS=3 Cao Nhất (Bỏ Trốn)", "Residual vs % Di Cư (Scatter)"),
        horizontal_spacing=0.12)
    fig3.add_trace(go.Bar(
        y=mgs_top['TINH'], x=mgs_top['MGS3_%'], orientation='h',
        marker_color='#EF4444', text=[f"{v:.1f}%" for v in mgs_top['MGS3_%']], textposition='outside',
        showlegend=False
    ), row=1, col=1)
    fig3.add_trace(go.Scatter(
        x=mgs_by_tinh['DI_CU_%'], y=mgs_by_tinh['RESIDUAL_%'],
        mode='markers+text', text=mgs_by_tinh['TINH'],
        textposition='top center', textfont=dict(size=9),
        marker=dict(
            size=mgs_by_tinh['EAD_TY'].clip(1) * 3 + 8,
            color=mgs_by_tinh['RESIDUAL_%'], colorscale='RdYlGn',
            showscale=True, colorbar=dict(title='Residual %', x=1.02)
        ),
        hovertemplate="<b>%{text}</b><br>Di Cư %: %{x:.1f}%<br>Residual: %{y:.2f}%<extra></extra>",
        showlegend=False
    ), row=1, col=2)
    fig3.add_hline(y=0, line_dash="dash", line_color="#FDD835", row=1, col=2)
    fig3.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=50, l=40, r=100))
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # ─── HTML Tables ────────────────────────────────────────────
    # Province full table (from tinh_agg)
    tinh_table_rows = ""
    for _, r in tinh_agg.sort_values('RESIDUAL_TB').iterrows():
        res_color = "#EF4444" if r['RESIDUAL_TB'] < -2 else ("#F59E0B" if r['RESIDUAL_TB'] < 0 else "#10B981")
        tinh_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['TINH']}</td>
            <td style="text-align:right;">{r['SỐ_HỒ_SƠ']:,}</td>
            <td style="text-align:right;">{r['TỶ_LỆ_THỰC_TẾ']:.1f}%</td>
            <td style="text-align:right;">{r['TỶ_LỆ_DỰ_BÁO']:.1f}%</td>
            <td style="text-align:right; font-weight:700; color:{res_color};">{r['RESIDUAL_TB']:+.3f}%</td>
        </tr>"""
    # MGS ranking table
    mgs_table_rows = ""
    for _, r in mgs_by_tinh.iterrows():
        mgs_color = "#EF4444" if r['MGS3_%'] > 20 else ("#F59E0B" if r['MGS3_%'] > 10 else "#10B981")
        mgs_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['TINH']}</td>
            <td style="text-align:right;">{r['SO_HS']:,}</td>
            <td style="text-align:right; font-weight:700; color:{mgs_color};">{r['MGS3_%']:.1f}%</td>
            <td style="text-align:right;">{r['MGS2_CT']:,}</td>
            <td style="text-align:right;">{r['DI_CU_%']:.1f}%</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right;">{r['RESIDUAL_%']:+.3f}%</td>
        </tr>"""

    # Insights HTML
    insights_html = ""
    top_worst = worst.iloc[0]['TINH'] if not worst.empty else "N/A"
    insights_html += f"""<div class="alert-box alert-danger"><strong>🔴 Điểm nóng khó đòi ({top_worst}):</strong> Tỉnh này có mức thu hồi thấp hơn mô hình dự báo đáng kể. Yêu cầu rà soát lại năng lực Agency tại địa phương hoặc chuyển giao Pháp lý sớm.</div>"""
    
    if skip_count > 0:
        insights_html += f"""<div class="alert-box alert-warn"><strong>⚠️ Rủi ro Skip-Trace tối cao:</strong> Phát hiện {skip_count:,} hồ sơ có dấu hiệu bỏ trốn (Migration Gap Score = 3: Cờ di cư + Lệch địa chỉ + Việc làm rủi ro). Cần ưu tiên đội đặc nhiệm xác minh hiện trường (Field Visit) cho nhóm này.</div>"""
    
    insights_html += f"""<div class="alert-box alert-success"><strong>✅ Mức độ tin cậy mô hình:</strong> Residual đã loại trừ độ khó của DPD và Nợ gốc. Do đó, các tỉnh có residual âm là thực sự kém hiệu quả do yếu tố ngoại cảnh (kinh tế địa phương, văn hóa, hoặc Agency).</div>"""

    total_loans = len(agg)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8D — Geographic Residual Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg: #F1F5F9; --card: #F8FAFC; --primary: #0F172A; --success: #10B981;
    --danger: #EF4444; --warn: #F59E0B; --muted: #64748B; --border: #E2E8F0;
    --text: #0F172A; --radius: 12px;
}}
[data-theme="dark"] {{
    --bg: #0F172A; --card: #1E293B; --primary: #38BDF8; --border: #334155;
    --text: #F1F5F9; --muted: #94A3B8;
}}
body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); padding: 24px; margin: 0; }}
.header {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 24px; }}
.header h1 {{ margin: 0 0 8px 0; font-size: 20px; font-weight: 800; color: var(--primary); letter-spacing: -0.5px; }}
.header p {{ margin: 0; font-size: 13px; color: var(--muted); }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.kpi-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; position: relative; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); }}
.kpi-card::after {{ content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: var(--primary); border-radius: 0 0 var(--radius) var(--radius); }}
.kpi-card.kpi-success::after {{ background: var(--success); }}
.kpi-card.kpi-danger::after {{ background: var(--danger); }}
.kpi-card.kpi-warn::after {{ background: var(--warn); }}
.kpi-label {{ font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 24px; font-weight: 800; line-height: 1.1; }}
.kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); margin-bottom: 0; }}
.tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; flex-wrap: wrap; }}
.data-table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.data-table th {{ background: #0F172A; padding: 10px 12px; font-weight: 700; color: var(--primary); border-bottom: 2px solid var(--border); text-align: left; white-space: nowrap; }}
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
    <h1>8D — PHÂN TÍCH ĐỊA LÝ DỰA TRÊN PHẦN DƯ (RESIDUAL)</h1>
    <p>Đo lường mức độ khó đòi thực sự của từng tỉnh (đã loại trừ nhiễu DPD và Nợ gốc) và dự báo rủi ro bỏ trốn (Skip-tracing) — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Số LOAN ID</div>
        <div class="kpi-value">{total_loans:,}</div>
        <div class="kpi-sub">Đã xác định địa chỉ</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Skip-Trace Tối Cao</div>
        <div class="kpi-value">{skip_count:,}</div>
        <div class="kpi-sub">Cần truy tìm hiện trường gấp</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Tỉnh Khó Đòi Nhất</div>
        <div class="kpi-value">{top_worst}</div>
        <div class="kpi-sub">Residual âm sâu nhất</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Tỉnh Dễ Thu Hồi Nhất</div>
        <div class="kpi-value">{best.iloc[-1]['TINH'] if not best.empty else 'N/A'}</div>
        <div class="kpi-sub">Residual dương cao nhất</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Residual Theo Tỉnh</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. Phân Tích Vùng Miền 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. MGS Ranking & Scatter 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Bảng Dữ Liệu Chi Tiết 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Insight & Skip-Tracing AI</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="chart-card">{div1}</div>
</div>

<div id="tab2" class="tab-content">
    <div class="chart-card">{div2}</div>
</div>

<div id="tab3" class="tab-content">
    <div class="chart-card">{div3}</div>
</div>

<div id="tab4" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 Bảng Residual Tất Cả Tỉnh (Sắp Xếp Từ Khó → Dễ)</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Tỉnh/Thành</th><th>Số HS</th><th>Tỷ Lệ Thực Tế</th>
                <th>Dự Báo Baseline</th><th>Residual (TT-DB)</th>
            </tr></thead>
            <tbody>{tinh_table_rows}</tbody>
        </table></div>
    </div>
    <div class="chart-card" style="margin-top:16px;">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">🔍 Bảng Migration Gap Score Theo Tỉnh</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Tỉnh/Thành</th><th>Số HS</th><th>% MGS=3</th><th>MGS=2 (ct)</th>
                <th>% Di Cư</th><th>EAD (Tỷ)</th><th>Residual</th>
            </tr></thead>
            <tbody>{mgs_table_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Chiến Lược Địa Phương Khóa & Truy Tìm</h3>
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
    
    // Initial plotly theme adjustment and resize trigger on DOMContentLoaded
    window.addEventListener('DOMContentLoaded', () => {{
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        setTimeout(() => {{ 
            updatePlotlyTheme(isDark); 
            window.dispatchEvent(new Event('resize'));
        }}, 300);
    }});
    // Safe fallback resize dispatch on window load
    window.addEventListener('load', () => {{
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 200);
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 800);
    }});
}})();
</script>
</body>
</html>"""

    out_html = os.path.join(REPORT_DIR, "8d_GEO_RESIDUAL.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv = os.path.join(SUB_DATA_DIR, "8d_geo_residual.csv")
    tinh_agg.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8D!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")

if __name__ == "__main__":
    run()
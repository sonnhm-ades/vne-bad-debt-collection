# -*- coding: utf-8 -*-
"""
MODULE 8E — PHÂN TÍCH ĐƯỜNG CONG VINTAGE THEO SẢN PHẨM + VELOCITY (PRODUCT VINTAGE CURVE + RECOVERY VELOCITY)
Phiên bản: 2.0 (2026-05-26) — Tích hợp SỐ_NGÀY_KHÔNG_THANH_TOÁN, TUOI, PTP signal
Câu hỏi: Loại sản phẩm nào suy giảm khả năng thu hồi nhanh nhất? Velocity thu hồi có khác nhau không?
Biến nguồn (CLEANED.csv + DS_SEGMENTATION_FINAL.csv):
  - KẾT QUẢ, SẢN PHẨM, NGÀY GIẢI NGÂN, NỢ GỐC (CLEANED)
  - SỐ_NGÀY_KHÔNG_THANH_TOÁN, TỔNG ĐÃ THANH TOÁN, TUỔI, PTP_SCORE_PERCENT (DS_SEG)
Output:  reports/Data_Science/PRODUCT_VINTAGE.html
"""
import pandas as pd
import numpy as np
import os, sys, warnings
import plotly.graph_objects as go
import plotly.express as px
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

FILE_PATH   = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'
DS_SEG_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science\Data\DS_SEGMENTATION_FINAL.csv'

MIN_LOANS_PER_CELL = 50   # Ô vintage cần ít nhất n hồ sơ

# ── Bảng Phân Nhóm Sản Phẩm (Rule-based keyword mapping) ────────────────────
PRODUCT_GROUPS = {
    'CASH LOAN': ['cash', 'i.cash', 'tiền mặt', 'personal loan', 'money'],
    'SHOPPING':  ['shopping', 'mua sắm', 'shop', 'loan_purpose_shopping', 'retail'],
    'VEHICLE':   ['vehicle', 'xe', 'motor', 'loan_purpose_vehicle', 'auto', 'oto'],
    'SALPIL':    ['salpil', 'sal'],
    'DRS/TLS':   ['drs', 'tls'],
}
DEFAULT_GROUP = 'Khác'

def classify_product(name):
    if pd.isna(name): return DEFAULT_GROUP
    n = str(name).lower().strip()
    for group, keywords in PRODUCT_GROUPS.items():
        if any(k in n for k in keywords):
            return group
    return DEFAULT_GROUP

def run():
    print("=" * 60)
    print("MODULE 8E — PRODUCT VINTAGE CURVE ANALYSIS")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────
    print("\n[1/5] Nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ']       = pd.to_numeric(df.get('KẾT QUẢ'),       errors='coerce').fillna(0)
    df['NGÀY GIẢI NGÂN'] = pd.to_datetime(df.get('NGÀY GIẢI NGÂN'), errors='coerce')

    # ── 2. Long → Wide per LOAN ID ──────────────────────────
    print("[2/5] Gom nhóm LOAN ID...")
    agg = df.groupby('LOAN ID').agg(
        KQ_TONG    = ('KẾT QUẢ',        'sum'),
        SP         = ('SẢN PHẨM',       'first'),
        NGAY_GN    = ('NGÀY GIẢI NGÂN', 'first'),
        NO_GOC     = ('NỢ GỐC',         'last'),
        TONG_DA_TT = ('TỔNG ĐÃ THANH TOÁN', 'last'),
    ).reset_index()

    df['NỢ GỐC'] = pd.to_numeric(df.get('NỢ GỐC'), errors='coerce').fillna(0)

    # Merge DS_SEG cho SỐ_NGÀY, TUỔI, PTP
    print("   Nạp DS_SEGMENTATION_FINAL.csv...")
    try:
        seg_chk = pd.read_csv(DS_SEG_PATH, nrows=1, low_memory=False)
        seg_want = ['LOAN ID', 'SỐ_NGÀY_KHÔNG_THANH_TOÁN', 'TUỔI',
                    'PTP_SCORE_PERCENT', 'CỤM_HÀNH_VI', 'SỐ KỲ ĐÃ TT']
        seg_use = [c for c in seg_want if c in seg_chk.columns]
        seg_df  = pd.read_csv(DS_SEG_PATH, low_memory=False, usecols=seg_use)
        seg_df  = seg_df.drop_duplicates(subset='LOAN ID', keep='last')
        agg = agg.merge(seg_df, on='LOAN ID', how='left')
        print(f"   → Đã gắn {len(seg_use)-1} cột từ DS_SEG")
    except Exception as e:
        print(f"   ⚠ Không tải DS_SEG: {e}")

    agg['CÓ_THU_TIỀN'] = (agg['KQ_TONG'] > 0).astype(int)
    agg['NHÓM_SP']     = agg['SP'].apply(classify_product)
    agg = agg.dropna(subset=['NGAY_GN'])

    # Vintage = Năm-Quý giải ngân
    agg['VINTAGE'] = agg['NGAY_GN'].dt.to_period('Q').astype(str)

    print(f"   → {len(agg):,} LOAN ID; {agg['NHÓM_SP'].nunique()} nhóm sản phẩm")
    print("   Phân bổ nhóm SP:")
    print(agg['NHÓM_SP'].value_counts().to_string())

    # ── 3. Tính Recovery Rate theo Vintage × Sản phẩm ─────────
    print("\n[3/5] Tính Recovery Rate theo Vintage × Sản phẩm...")
    pivot = agg.groupby(['VINTAGE', 'NHÓM_SP']).agg(
        TỔNG_HS       = ('LOAN ID',      'count'),
        TỔNG_ĐÃ_TRẢ   = ('CÓ_THU_TIỀN', 'sum'),
    ).reset_index()
    pivot = pivot[pivot['TỔNG_HS'] >= MIN_LOANS_PER_CELL]
    pivot['TỶ_LỆ_%'] = (pivot['TỔNG_ĐÃ_TRẢ'] / pivot['TỔNG_HS'] * 100).round(2)

    # ── 4. Tổng hợp tỷ lệ NPL & Recovery theo nhóm SP ─────────
    # ── 4. Tổng hợp tỷ lệ NPL & Recovery theo nhóm SP ─────────
    print("[4/5] Tổng hợp theo nhóm SP...")
    sp_summary = agg.groupby('NHÓM_SP').agg(
        TỔNG_HS        = ('LOAN ID',      'count'),
        TỶ_LỆ_THU_HỒI  = ('CÓ_THU_TIỀN', 'mean'),
        NỢ_GỐC_TB      = ('NO_GOC',       'mean'),
        EAD            = ('NO_GOC',       'sum'),
        KQ_TONG        = ('KQ_TONG',      'sum')
    ).reset_index()
    sp_summary['TỶ_LỆ_THU_HỒI_%'] = (sp_summary['TỶ_LỆ_THU_HỒI'] * 100).round(2)
    sp_summary['EAD_TY']          = (sp_summary['EAD'] / 1e9).round(2)
    sp_summary['THU_TY']          = (sp_summary['KQ_TONG'] / 1e9).round(3)
    sp_summary['NỢ_GỐC_TB']       = sp_summary['NỢ_GỐC_TB'].round(0)
    sp_summary = sp_summary.sort_values('TỶ_LỆ_THU_HỒI_%', ascending=False)
    print(sp_summary[['NHÓM_SP', 'TỔNG_HS', 'TỶ_LỆ_THU_HỒI_%', 'EAD_TY', 'THU_TY']].to_string(index=False))

    # ── 4B. Tốc độ thu hồi (Recovery Velocity) ─────────
    # Giả định "TỐC ĐỘ" đo lường bằng tỷ lệ trả trong < 90 DPD so với tổng trả
    print("[4b/5] Tính Recovery Velocity...")
    agg['EARLY_PAY'] = ((agg['CÓ_THU_TIỀN'] == 1) & (agg['SỐ_NGÀY_KHÔNG_THANH_TOÁN'] <= 90)).astype(int) if 'SỐ_NGÀY_KHÔNG_THANH_TOÁN' in agg.columns else 0
    sp_velocity = agg[agg['CÓ_THU_TIỀN'] == 1].groupby('NHÓM_SP').agg(
        TOTAL_PAID = ('LOAN ID', 'count'),
        EARLY_PAID = ('EARLY_PAY', 'sum')
    ).reset_index()
    sp_velocity['VELOCITY_%'] = (sp_velocity['EARLY_PAID'] / sp_velocity['TOTAL_PAID'] * 100).round(2)

    # ── 5. Visualization ─────────────────────────────────────────
    print("\n[5/5] Tạo Dashboard HTML cao cấp...")
    import plotly.offline as plo

    all_groups = agg['NHÓM_SP'].value_counts().index.tolist()
    palette    = px.colors.qualitative.Bold

    # Tab 1: Vintage Line
    fig1 = go.Figure()
    for i, grp in enumerate(all_groups):
        grp_data = pivot[pivot['NHÓM_SP'] == grp].sort_values('VINTAGE')
        if len(grp_data) < 2: continue
        fig1.add_trace(go.Scatter(
            x=grp_data['VINTAGE'], y=grp_data['TỶ_LỆ_%'],
            mode='lines+markers', name=grp,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{grp}</b><br>Quý: %{{x}}<br>Tỷ lệ: %{{y:.2f}}%<extra></extra>",
        ))
    fig1.update_layout( height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40), xaxis_title="Quý Giải Ngân", yaxis_title="Tỷ lệ thu hồi (%)")
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Tab 2: Hiệu Quả Nhóm Sản Phẩm
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Tỷ Lệ Thu Hồi (%)", "Phân Bổ Hồ Sơ"), specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig2.add_trace(go.Bar(
        x=sp_summary['NHÓM_SP'], y=sp_summary['TỶ_LỆ_THU_HỒI_%'],
        marker_color=[palette[i % len(palette)] for i in range(len(sp_summary))],
        text=[f"{v:.2f}%" for v in sp_summary['TỶ_LỆ_THU_HỒI_%']],
        textposition='outside', showlegend=False,
    ), row=1, col=1)
    fig2.add_trace(go.Pie(
        labels=sp_summary['NHÓM_SP'], values=sp_summary['TỔNG_HS'],
        hole=0.4,
        marker_colors=[palette[i % len(palette)] for i in range(len(sp_summary))],
        textinfo='label+percent', showlegend=False,
    ), row=1, col=2)
    fig2.update_layout( height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # Tab 3: Recovery Velocity & EAD
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Quy mô EAD (Tỷ VND) Theo Sản Phẩm", "Tốc Độ Thu Hồi (<90 DPD) Trong Nhóm Đã Trả"), specs=[[{"type": "bar"}, {"type": "bar"}]], horizontal_spacing=0.1)
    
    fig3.add_trace(go.Bar(
        x=sp_summary['NHÓM_SP'], y=sp_summary['EAD_TY'],
        marker_color='#1E88E5', text=[f"{v:.1f}T" for v in sp_summary['EAD_TY']], textposition='outside'
    ), row=1, col=1)
    
    if not sp_velocity.empty:
        sp_velocity = sp_velocity.sort_values('VELOCITY_%', ascending=False)
        fig3.add_trace(go.Bar(
            x=sp_velocity['NHÓM_SP'], y=sp_velocity['VELOCITY_%'],
            marker_color='#F59E0B', text=[f"{v:.1f}%" for v in sp_velocity['VELOCITY_%']], textposition='outside'
        ), row=1, col=2)
    
    fig3.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=50, l=40, r=40))
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # Tab 4: Data Tables
    table_rows = ""
    for _, r in sp_summary.iterrows():
        rate_color = "#10B981" if r['TỶ_LỆ_THU_HỒI_%'] > 5 else ("#F59E0B" if r['TỶ_LỆ_THU_HỒI_%'] > 2 else "#EF4444")
        velocity_val = sp_velocity.loc[sp_velocity['NHÓM_SP'] == r['NHÓM_SP'], 'VELOCITY_%'].values
        velocity_str = f"{velocity_val[0]:.1f}%" if len(velocity_val) > 0 else "N/A"
        table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['NHÓM_SP']}</td>
            <td style="text-align:right;">{r['TỔNG_HS']:,}</td>
            <td style="text-align:right; font-weight:700; color:{rate_color};">{r['TỶ_LỆ_THU_HỒI_%']:.2f}%</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right; color:#10B981;">{r['THU_TY']:.3f} Tỷ</td>
            <td style="text-align:right;">{velocity_str}</td>
            <td style="text-align:right;">{r['NỢ_GỐC_TB']/1e6:.1f}M</td>
        </tr>"""

    # Insights HTML
    best_sp = sp_summary.iloc[0]
    worst_sp = sp_summary.iloc[-1]
    
    insights_html = f"""<div class="alert-box alert-success"><strong>⭐ Quán Quân Thu Hồi:</strong> Sản phẩm <b>{best_sp['NHÓM_SP']}</b> đạt tỷ lệ {best_sp['TỶ_LỆ_THU_HỒI_%']:.2f}% (Tốt nhất). Đây là mỏ neo tạo Cash-Flow cho tháng tới, cần dồn nguồn lực Telesales ưu tiên.</div>"""
    
    insights_html += f"""<div class="alert-box alert-danger"><strong>🔴 Cảnh Báo Đỏ:</strong> Nhóm <b>{worst_sp['NHÓM_SP']}</b> có tỷ lệ thu hồi thấp nhất ({worst_sp['TỶ_LỆ_THU_HỒI_%']:.2f}%). Cần giảm thời gian gọi điện và cân nhắc chuyển sớm sang quy trình Pháp Lý.</div>"""
    
    insights_html += f"""<div class="alert-box alert-warn"><strong>⚠️ Vintage Decay:</strong> Xu hướng (Velocity) thu hồi qua các quý giải ngân cho thấy rủi ro suy giảm chất lượng khoản vay. Cần theo dõi đường cong Vintage của {worst_sp['NHÓM_SP']} để nhận diện điểm đứt gãy.</div>"""

    total_loans = len(agg)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8E — Product Vintage Analysis</title>
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
    <h1>8E — VINTAGE CỦA SẢN PHẨM & TỐC ĐỘ THU HỒI</h1>
    <p>Phân tích xu hướng chất lượng khoản vay theo thời gian (Vintage Curve) và hiệu suất thu hồi của từng dòng sản phẩm — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Hồ Sơ</div>
        <div class="kpi-value">{total_loans:,}</div>
        <div class="kpi-sub">LOAN ID hợp lệ</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Sản Phẩm Tốt Nhất</div>
        <div class="kpi-value">{best_sp['NHÓM_SP']}</div>
        <div class="kpi-sub">RR: {best_sp['TỶ_LỆ_THU_HỒI_%']:.2f}%</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Sản Phẩm Tệ Nhất</div>
        <div class="kpi-value">{worst_sp['NHÓM_SP']}</div>
        <div class="kpi-sub">RR: {worst_sp['TỶ_LỆ_THU_HỒI_%']:.2f}%</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Vintage Curve</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. Hiệu Quả Sản Phẩm</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. Recovery Velocity 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Bảng Chi Tiết Sản Phẩm 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Chiến Lược Danh Mục</button>
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
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 Phân Tích Danh Mục Sản Phẩm (Nợ Gốc & Tiền Thu)</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Nhóm Sản Phẩm</th><th>Số Hồ Sơ</th><th>Tỷ Lệ Thu Hồi</th>
                <th>EAD (Tỷ)</th><th>Tiền Thu (Tỷ)</th><th>Recovery Velocity (&lt;90 DPD)</th><th>Nợ Gốc TB</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Chiến Lược Quản Trị Danh Mục</h3>
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

    out_html = os.path.join(REPORT_DIR, "8e_PRODUCT_VINTAGE.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv  = os.path.join(SUB_DATA_DIR, "8e_product_vintage.csv")
    pivot.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8E!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")

if __name__ == "__main__":
    run()
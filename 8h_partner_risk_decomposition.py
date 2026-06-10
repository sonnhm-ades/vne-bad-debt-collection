# -*- coding: utf-8 -*-
"""
MODULE 8H — PARTNER RISK DECOMPOSITION (Rủi Ro Tập Trung Theo Đối Tác)
Câu hỏi: Đối tác nào đang gánh tập trung rủi ro? Hồ sơ rotate nhiều lần có kết quả tệ hơn không?
Output:  reports/Data_Science/Reports/8h_PARTNER_RISK.html + 8h_partner_risk.csv
"""
import pandas as pd
import numpy as np
import os, sys, warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import f_oneway

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
LGD_PATH   = os.path.join(SUB_DATA_DIR, '7d_lgd_results.csv')

def hhi(series):
    """Herfindahl-Hirschman Index — đo mức độ tập trung (0=hoàn toàn phân tán, 1=độc quyền)"""
    total = series.sum()
    if total == 0: return 0
    shares = series / total
    return (shares ** 2).sum()

def run():
    print("=" * 60)
    print("MODULE 8H — PARTNER RISK DECOMPOSITION")
    print("=" * 60)

    # ── 1. Load raw + LGD results ─────────────────────────────
    print("\n[1/5] Nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df['KẾT QUẢ'], errors='coerce').fillna(0)
    df['NỢ GỐC']  = pd.to_numeric(df['NỢ GỐC'],  errors='coerce').fillna(0)
    df['DPD']     = pd.to_numeric(df['DPD'],     errors='coerce')

    lgd = pd.read_csv(LGD_PATH, encoding='utf-8-sig')
    lgd_map = lgd.set_index('LOAN ID')[['EL_VND','PD_SCORE','LGD_PRED','REC_RATE']].to_dict('index')

    # ── 2. Aggregate per LOAN ID ──────────────────────────────
    print("[2/5] Gom nhóm LOAN ID ...")
    agg = df.groupby('LOAN ID').agg(
        KQ_TONG      = ('KẾT QUẢ',              'sum'),
        NO_GOC       = ('NỢ GỐC',               'last'),
        DPD          = ('DPD',                  'last'),
        DU_AN        = ('DỰ ÁN',                'first'),
        DOI_TAC_PL1  = ('ĐỐI TÁC PL1',         'first'),
        DOI_TAC_PL2  = ('ĐỐI TÁC PL2',         'first'),
        RATING       = ('ĐÁNH GIÁ KHÁCH HÀNG', 'last'),
        LAI_SUAT     = ('LÃI SUẤT',             'max'),
        TINH         = ('TỈNH TẠM TRÚ',         'first'),
    ).reset_index()

    agg['CÓ_TRẢ'] = (agg['KQ_TONG'] > 0).astype(int)
    # Merge EL
    agg['EL_VND']   = agg['LOAN ID'].map(lambda x: lgd_map.get(x, {}).get('EL_VND', agg['NO_GOC'].get(0,0)))
    agg['PD_SCORE'] = agg['LOAN ID'].map(lambda x: lgd_map.get(x, {}).get('PD_SCORE', 0.99))

    # ── 3. Portfolio Concentration (HHI) ─────────────────────
    print("[3/5] Tính Concentration Risk (HHI) ...")
    by_partner = agg.groupby('DU_AN').agg(
        TONG_EAD    = ('NO_GOC',   'sum'),
        SO_HO_SO    = ('LOAN ID',  'count'),
        RATE_THU_HOI= ('CÓ_TRẢ',  'mean'),
        AVG_DPD     = ('DPD',      'mean'),
        RATING_A    = ('RATING',   lambda x: (x=='A').mean()),
    ).reset_index()
    by_partner['EAD_BLN'] = (by_partner['TONG_EAD'] / 1e9).round(1)
    by_partner['RATE_%']  = (by_partner['RATE_THU_HOI'] * 100).round(2)
    by_partner['RATING_A_%'] = (by_partner['RATING_A'] * 100).round(1)
    by_partner['AVG_DPD'] = by_partner['AVG_DPD'].round(0)

    hhiEAD = hhi(by_partner['TONG_EAD'])
    hhiHS  = hhi(by_partner['SO_HO_SO'])
    risk_level = "🔴 CAO" if hhiEAD > 0.25 else ("🟡 TRUNG BÌNH" if hhiEAD > 0.15 else "🟢 THẤP")
    
    # Calculate EL rate by partner (using LGD data)
    el_by_partner = agg.groupby('DU_AN').agg(
        TONG_EAD = ('NO_GOC', 'sum'),
        TONG_EL = ('EL_VND', 'sum')
    )
    by_partner['EL_RATE'] = el_by_partner['TONG_EL'].values / el_by_partner['TONG_EAD'].values
    by_partner['EL_RATE_%'] = (by_partner['EL_RATE'] * 100).round(2)
    
    # Insights Generator Logic
    insights_html = ""
    if hhiEAD > 0.25:
        top_partner = by_partner.sort_values('TONG_EAD', ascending=False).iloc[0]['DU_AN']
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>🔴 Rủi Ro Tập Trung EAD CAO (HHI > 0.25):</strong> Danh mục bị phụ thuộc lớn vào một vài đối tác, đặc biệt là <b>{top_partner}</b>. Khuyến nghị phân bổ lại room giải ngân/thu hồi cho các đối tác khác để giảm rủi ro tập trung.
        </div>"""
    elif hhiEAD > 0.15:
        insights_html += """
        <div class="alert-box alert-warn">
            <strong>🟡 Rủi Ro Tập Trung EAD TRUNG BÌNH:</strong> Có dấu hiệu tập trung vốn, cần theo dõi thêm chỉ số HHI trong kỳ tới.
        </div>"""
    else:
        insights_html += """
        <div class="alert-box alert-success">
            <strong>🟢 Rủi Ro Tập Trung THẤP:</strong> Danh mục được phân tán tốt giữa các đối tác. Mức độ rủi ro hệ thống từ việc phụ thuộc đối tác là thấp.
        </div>"""
        
    high_el_partners = by_partner[by_partner['EL_RATE'] > 0.8]['DU_AN'].tolist()
    if len(high_el_partners) > 0:
        partners_str = ", ".join(high_el_partners)
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>⚠️ Đối tác vượt ngưỡng rủi ro Tổn thất kỳ vọng:</strong> Các đối tác {partners_str} có Tỷ lệ Tổn thất Kỳ vọng (EL Rate) > 80%. Khuyến nghị đánh giá lại tiêu chuẩn và quy trình xử lý nợ từ các nguồn này.
        </div>"""

    # Stress Testing (Giả định Recovery Rate giảm)
    stress_results = []
    current_total_el = agg['EL_VND'].sum()
    for shock in [0.05, 0.10, 0.20]:  # Giảm REC_RATE 5%, 10%, 20%
        # EL_new = EL_old + PD * shock * EAD
        stress_el = current_total_el + (agg['PD_SCORE'] * shock * agg['NO_GOC']).sum()
        stress_results.append({
            'Kịch_bản': f'Giảm {int(shock*100)}% Thu Hồi',
            'EL_Tăng_Thêm_Tỷ': (stress_el - current_total_el) / 1e9,
            'Tổng_EL_Mới_Tỷ': stress_el / 1e9
        })
    stress_df = pd.DataFrame(stress_results)

    print(f"\n   HHI theo EAD: {hhiEAD:.4f} — {risk_level}")
    print(f"   HHI theo HS:  {hhiHS:.4f}")
    print("\n   EAD & Recovery Rate theo Đối Tác:")
    print(by_partner[['DU_AN','EAD_BLN','SO_HO_SO','RATE_%','EL_RATE_%','AVG_DPD']].sort_values('EAD_BLN', ascending=False).to_string(index=False))

    # ── 4. Rotation Analysis (ĐỐI TÁC PL2) ──────────────────
    print("\n[4/5] Phân tích Rotation của hồ sơ (ĐỐI TÁC PL2) ...")
    # Đếm số lần rotate: ROTATE.01.26, ROTATE.03.26... → extract month count
    def count_rotate(val):
        if pd.isna(val): return 0
        s = str(val).upper()
        if 'ROTATE' in s:
            # Extract number from format ROTATE.MM.YY → count how many months back
            return 1
        return 0

    agg['IS_ROTATED'] = agg['DOI_TAC_PL2'].apply(count_rotate)
    agg['IS_NEW']     = (agg['DOI_TAC_PL1'] == 'NEW').astype(int)

    rotate_vs_new = agg.groupby('IS_ROTATED').agg(
        SO_HS    = ('LOAN ID',  'count'),
        RATE     = ('CÓ_TRẢ',  'mean'),
        AVG_DPD  = ('DPD',     'mean'),
        AVG_EAD  = ('NO_GOC',  'mean'),
    ).reset_index()
    rotate_vs_new['IS_ROTATED'] = rotate_vs_new['IS_ROTATED'].map({0: 'Hồ Sơ Mới (NEW)', 1: 'Hồ Sơ Rotated'})
    rotate_vs_new['RATE_%']     = (rotate_vs_new['RATE'] * 100).round(2)
    print(rotate_vs_new[['IS_ROTATED','SO_HS','RATE_%','AVG_DPD']].to_string(index=False))

    # Recovery by ROTATION SOURCE
    rot_source = agg.groupby('DOI_TAC_PL2').agg(
        SO_HS = ('LOAN ID','count'),
        RATE  = ('CÓ_TRẢ','mean'),
        AVG_DPD = ('DPD','mean'),
    ).reset_index()
    rot_source = rot_source[rot_source['SO_HS'] >= 500]
    rot_source['RATE_%'] = (rot_source['RATE'] * 100).round(2)
    rot_source = rot_source.sort_values('RATE_%', ascending=False)
    print("\n   Top Rotation Batches (≥500 HS):")
    print(rot_source.head(10)[['DOI_TAC_PL2','SO_HS','RATE_%','AVG_DPD']].to_string(index=False))

    # ANOVA: Recovery rate có khác nhau đáng kể giữa các đối tác không?
    anova_groups = [agg[agg['DU_AN']==p]['CÓ_TRẢ'].values for p in by_partner['DU_AN']]
    anova_groups = [g for g in anova_groups if len(g) > 10]
    try:
        f_stat, p_val = f_oneway(*anova_groups)
        sig = "✅ CÓ Ý NGHĨA THỐNG KÊ (p<0.05)" if p_val < 0.05 else "❌ Không đủ ý nghĩa thống kê"
        print(f"\n   ANOVA: F={f_stat:.2f}, p={p_val:.6f} → {sig}")
    except Exception as e:
        p_val, f_stat = None, None
        print(f"   ANOVA error: {e}")

    # ── 5. Visualization & Premium Layout ─────────────────────────────────────
    print("\n[5/5] Tạo Dashboard HTML cao cấp...")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Phân Bổ EAD Theo Đối Tác (Tỷ VND)",
            "Tỷ Lệ Thu Hồi Theo Đối Tác (%)",
            "Hiệu Quả Thu Hồi: Hồ Sơ Mới vs Rotated",
            "Tỷ Lệ Thu Hồi Theo Nhóm Rotation (PL2)",
        ),
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )

    # Plot 1: EAD by Partner
    by_partner_sorted = by_partner.sort_values('TONG_EAD', ascending=False)
    fig.add_trace(go.Bar(
        x=by_partner_sorted['DU_AN'],
        y=by_partner_sorted['EAD_BLN'],
        marker_color='#38BDF8',
        name='EAD (Tỷ VND)',
        text=by_partner_sorted['EAD_BLN'].apply(lambda v: f"{v:.1f}T"),
        textposition='outside',
        showlegend=False
    ), row=1, col=1)

    # Plot 2: Recovery Rate by Partner
    fig.add_trace(go.Bar(
        x=by_partner_sorted['DU_AN'],
        y=by_partner_sorted['RATE_%'],
        marker_color='#10B981',
        name='Paid Rate (%)',
        text=by_partner_sorted['RATE_%'].apply(lambda v: f"{v:.1f}%"),
        textposition='outside',
        showlegend=False
    ), row=1, col=2)

    # Plot 3: New vs Rotated Paid Rate
    fig.add_trace(go.Bar(
        x=rotate_vs_new['IS_ROTATED'],
        y=rotate_vs_new['RATE_%'],
        marker_color=['#10B981', '#F59E0B'],
        name='Paid Rate (%)',
        text=rotate_vs_new['RATE_%'].apply(lambda v: f"{v:.1f}%"),
        textposition='outside',
        showlegend=False
    ), row=2, col=1)

    # Plot 4: Top Rotation Batches
    top_rot = rot_source.head(8)
    fig.add_trace(go.Bar(
        x=top_rot['DOI_TAC_PL2'],
        y=top_rot['RATE_%'],
        marker_color='#EF4444',
        name='Paid Rate (%)',
        text=top_rot['RATE_%'].apply(lambda v: f"{v:.1f}%"),
        textposition='outside',
        showlegend=False
    ), row=2, col=2)

    # Update fig layout to dark mode
    fig.update_layout(
        
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=850,
        font=dict(family="Inter, sans-serif"),
        margin=dict(t=50, b=80, l=40, r=40),
    )
    # Remove annotations as we have HTML header
    fig.layout.annotations = []

    # Extract KPI metrics before charting
    total_ead = agg['NO_GOC'].sum()
    total_loans = len(agg)
    hhi_val = hhiEAD
    hhi_status = risk_level
    anova_sig_txt = sig if 'sig' in locals() else "Không tính toán"

    # Get HTML Plotly div for Main Chart
    import plotly.offline as plo
    chart_div = plo.plot(fig, output_type='div', include_plotlyjs=False)

    # Chart 2: Waterfall (Concentration Risk)
    waterfall_fig = go.Figure(go.Waterfall(
        name = "2026", orientation = "v",
        measure = ["relative"] * len(by_partner_sorted) + ["total"],
        x = by_partner_sorted['DU_AN'].tolist() + ["Tổng EAD"],
        textposition = "outside",
        text = [f"{v:.1f}T" for v in by_partner_sorted['EAD_BLN']] + [f"{total_ead/1e9:.1f}T"],
        y = by_partner_sorted['EAD_BLN'].tolist() + [0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    waterfall_fig.update_layout(
        title="Đóng Góp EAD Theo Từng Đối Tác (Waterfall Risk Concentration)",
         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, font=dict(family="Inter, sans-serif"), margin=dict(t=50, b=50, l=40, r=40)
    )
    waterfall_div = plo.plot(waterfall_fig, output_type='div', include_plotlyjs=False)

    # Chart 3: Stress Testing
    stress_fig = go.Figure(go.Bar(
        x=stress_df['Kịch_bản'],
        y=stress_df['EL_Tăng_Thêm_Tỷ'],
        marker_color='#EF4444',
        text=[f"+{v:.1f} Tỷ" for v in stress_df['EL_Tăng_Thêm_Tỷ']],
        textposition='outside'
    ))
    stress_fig.update_layout(
        title="Stress Test: Tăng Expected Loss Khi Tỷ Lệ Thu Hồi Giảm",
         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, font=dict(family="Inter, sans-serif"), margin=dict(t=50, b=50, l=40, r=40)
    )
    stress_div = plo.plot(stress_fig, output_type='div', include_plotlyjs=False)

    # Generate Data Tables HTML Rows
    partner_table_rows = ""
    for _, r in by_partner_sorted.iterrows():
        el_color = "#EF4444" if r['EL_RATE_%'] > 80 else "#10B981"
        partner_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['DU_AN']}</td>
            <td style="text-align:right;">{r['SO_HO_SO']:,}</td>
            <td style="text-align:right;">{r['EAD_BLN']:.1f} Tỷ</td>
            <td style="text-align:right;">{r['RATE_%']:.2f}%</td>
            <td style="text-align:right;">{r['AVG_DPD']:.0f}</td>
            <td style="text-align:right; font-weight:700; color:{el_color};">{r['EL_RATE_%']:.2f}%</td>
        </tr>"""

    rotate_table_rows = ""
    for _, r in rot_source.iterrows():
        rotate_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['DOI_TAC_PL2']}</td>
            <td style="text-align:right;">{r['SO_HS']:,}</td>
            <td style="text-align:right;">{r['RATE_%']:.2f}%</td>
            <td style="text-align:right;">{r['AVG_DPD']:.0f}</td>
        </tr>"""

    # Premium Layout HTML
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8H — Partner Concentration Risk</title>
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
.chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,.1);
}}
.tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
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
    <h1>8H — PARTNER RISK DECOMPOSITION</h1>
    <p>Phân tích rủi ro tập trung đối tác (Concentration Risk): Đánh giá mức độ tập trung danh mục EAD, so sánh kết quả thu hồi ANOVA và hiệu quả của hồ sơ NEW vs ROTATED — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Giá Trị EAD</div>
        <div class="kpi-value">{total_ead/1e9:,.1f} Tỷ</div>
        <div class="kpi-sub">Tổng nợ gốc danh mục</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Tổng Hồ Sơ</div>
        <div class="kpi-value">{total_loans:,} HS</div>
        <div class="kpi-sub">Số lượng hồ sơ đang quản lý</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Chỉ Số Tập Trung HHI</div>
        <div class="kpi-value">{hhi_val:.4f}</div>
        <div class="kpi-sub">Mức độ tập trung: <strong>{hhi_status}</strong></div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">ANOVA Test Signif.</div>
        <div class="kpi-value">{anova_sig_txt}</div>
        <div class="kpi-sub">Khác biệt thu hồi giữa các đối tác</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Tổng Quan Đối Tác</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. Phân Tích Tập Trung & Stress Test</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. Bảng Dữ Liệu Đối Tác 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Lịch Sử Xoay Vòng 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Insight & Khuyến Nghị</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="chart-card">{chart_div}</div>
</div>

<div id="tab2" class="tab-content">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div class="chart-card">{waterfall_div}</div>
        <div class="chart-card">{stress_div}</div>
    </div>
</div>

<div id="tab3" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 Phân Bổ Danh Mục Theo Đối Tác</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Đối Tác (PL1)</th><th>Số HS</th><th>EAD Tổng</th>
                <th>Tỷ Lệ Thu Hồi</th><th>DPD Trung Bình</th><th>Tỷ Lệ EL / EAD</th>
            </tr></thead>
            <tbody>{partner_table_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab4" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">🔄 Hiệu Quả Nguồn Gốc Hồ Sơ (PL2 / Rotation)</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Lịch Sử Xoay Vòng (PL2)</th><th>Số HS</th><th>Tỷ Lệ Thu Hồi</th><th>DPD Trung Bình</th>
            </tr></thead>
            <tbody>{rotate_table_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Khuyến Nghị Hành Động Dựa Trên Dữ Liệu</h3>
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

    out_html = os.path.join(REPORT_DIR, "8h_PARTNER_RISK.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    out_csv  = os.path.join(SUB_DATA_DIR, "8h_partner_risk.csv")
    by_partner.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8H!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")

if __name__ == "__main__":
    run()
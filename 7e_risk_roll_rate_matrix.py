# -*- coding: utf-8 -*-
"""
MODULE 7E — ROLL RATE MATRIX v3.0 (Deep Risk Manager Edition)
Câu hỏi: Portfolio đang "cải thiện" hay "xấu đi" qua 3 tháng? Bao nhiêu % hồ sơ leo thang DPD?
         Forward-looking: Dự báo Net Roll Rate tháng tới nếu xu hướng giữ nguyên?
         Cohort Analysis: Nhóm nào leo thang nhanh nhất?
Output:  reports/Data_Science/Reports/7e_ROLL_RATE.html + 7e_roll_rate.csv
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

FILE_PATH  = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'
MONTH_ORDER = ['THÁNG 01.26', 'THÁNG 02.26', 'THÁNG 03.26']

DPD_BANDS = [
    (0,    90,   "Current/Early (0–90)"),
    (91,   180,  "Fresh NPL (91–180)"),
    (181,  360,  "Khó Đòi Sớm (181–360)"),
    (361,  540,  "Khó Đòi GĐ1 (361–540)"),
    (541,  720,  "Khó Đòi GĐ2 (541–720)"),
    (721,  1080, "Khó Đòi GĐ3 (721–1080)"),
    (1081, 1440, "Nợ Sâu GĐ1 (1081–1440)"),
    (1441, 9999, "Nợ Sâu/TV (>1440)"),
]

def dpd_to_band(dpd):
    if pd.isna(dpd) or dpd < 0: return "Unknown"
    for lo, hi, label in DPD_BANDS:
        if lo <= dpd <= hi: return label
    return "Nợ Sâu/TV (>1440)"

def run():
    print("=" * 60)
    print("MODULE 7E — ROLL RATE MATRIX v3.0")
    print("=" * 60)

    # ── 1. Load ─────────────────────────────────────────────────
    print("\n[1/6] Nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['DPD']     = pd.to_numeric(df['DPD'], errors='coerce')
    df['KẾT QUẢ'] = pd.to_numeric(df['KẾT QUẢ'], errors='coerce').fillna(0)
    df['NỢ GỐC']  = pd.to_numeric(df['NỢ GỐC'], errors='coerce').fillna(0)

    # ── 2. Pivot ─────────────────────────────────────────────────
    print("[2/6] Pivot dữ liệu theo LOAN ID × THÁNG ...")
    pivot = df.pivot_table(index='LOAN ID', columns='THÁNG', values='DPD', aggfunc='last')
    pivot = pivot[[c for c in MONTH_ORDER if c in pivot.columns]]
    df_du_an   = df.groupby('LOAN ID')['DỰ ÁN'].first()
    df_no_goc  = df.groupby('LOAN ID')['NỢ GỐC'].last()
    df_rating  = df.groupby('LOAN ID')['ĐÁNH GIÁ KHÁCH HÀNG'].last()
    df_kq      = df.groupby('LOAN ID')['KẾT QUẢ'].sum()
    pivot = pivot.join(df_du_an).join(df_no_goc).join(df_rating).join(df_kq)
    print(f"   → {len(pivot):,} LOAN ID có ít nhất 1 tháng dữ liệu")

    for col in [c for c in pivot.columns if c not in ['DỰ ÁN','NỢ GỐC','ĐÁNH GIÁ KHÁCH HÀNG','KẾT QUẢ']]:
        pivot[f'{col}_BAND'] = pivot[col].apply(dpd_to_band)

    # ── 3. Roll Rate ─────────────────────────────────────────────
    print("[3/6] Tính Roll Rate giữa các cặp tháng ...")
    all_bands   = [b[2] for b in DPD_BANDS]
    results     = []
    transitions = {}
    band_order  = {b[2]: i for i, b in enumerate(DPD_BANDS)}

    month_pairs = [(MONTH_ORDER[i], MONTH_ORDER[i+1]) for i in range(len(MONTH_ORDER)-1)
                   if MONTH_ORDER[i] in pivot.columns and MONTH_ORDER[i+1] in pivot.columns]

    for m1, m2 in month_pairs:
        pair = pivot[[f'{m1}_BAND', f'{m2}_BAND', 'DỰ ÁN', 'NỢ GỐC', 'ĐÁNH GIÁ KHÁCH HÀNG']].dropna(subset=[f'{m1}_BAND', f'{m2}_BAND'])
        pair.columns = ['FROM_BAND', 'TO_BAND', 'DU_AN', 'NO_GOC', 'RATING']
        pair = pair[(pair['FROM_BAND'] != 'Unknown') & (pair['TO_BAND'] != 'Unknown')]

        cross = pd.crosstab(pair['FROM_BAND'], pair['TO_BAND'], normalize='index') * 100
        transitions[f"{m1}→{m2}"] = cross

        n_total   = len(pair)
        pair['FROM_IDX'] = pair['FROM_BAND'].map(band_order)
        pair['TO_IDX']   = pair['TO_BAND'].map(band_order)
        n_worse   = (pair['TO_IDX'] > pair['FROM_IDX']).sum()
        n_better  = (pair['TO_IDX'] < pair['FROM_IDX']).sum()
        n_stable  = (pair['TO_IDX'] == pair['FROM_IDX']).sum()
        net_roll  = (n_worse - n_better) / n_total * 100

        hard_debt_mask = pair['FROM_IDX'] > 0
        cured_mask = pair['TO_IDX'] == 0
        cure_rate = cured_mask[hard_debt_mask].mean() * 100 if hard_debt_mask.sum() > 0 else 0

        # NEW: EAD-weighted net roll
        total_ead = pair['NO_GOC'].sum()
        ead_worse  = pair.loc[pair['TO_IDX'] > pair['FROM_IDX'], 'NO_GOC'].sum()
        ead_better = pair.loc[pair['TO_IDX'] < pair['FROM_IDX'], 'NO_GOC'].sum()
        ead_net_roll = (ead_worse - ead_better) / total_ead * 100 if total_ead > 0 else 0

        results.append({
            'Cặp_Tháng':         f"{m1} → {m2}",
            'Tổng_HS':           n_total,
            'Forward_Roll_%':    round(n_worse  / n_total * 100, 2),
            'Recovery_Roll_%':   round(n_better / n_total * 100, 2),
            'Stable_%':          round(n_stable / n_total * 100, 2),
            'Net_Roll_%':        round(net_roll, 2),
            'EAD_Net_Roll_%':    round(ead_net_roll, 2),
            'Cure_Rate_%':       round(cure_rate, 2)
        })
        print(f"   {m1}→{m2}: Forward={n_worse/n_total*100:.1f}% | "
              f"Recovery={n_better/n_total*100:.1f}% | Net Roll={net_roll:.1f}% | "
              f"EAD Net Roll={ead_net_roll:.1f}% | Cure={cure_rate:.1f}%")

        if m1 == month_pairs[-1][0] and m2 == month_pairs[-1][1]:
            def calc_partner(x):
                t = len(x)
                if t == 0: return pd.Series({'Fwd':0, 'Rec':0, 'Net':0, 'Total':0, 'EAD':0})
                fw = (x['TO_IDX'] > x['FROM_IDX']).sum() / t * 100
                rc = (x['TO_IDX'] < x['FROM_IDX']).sum() / t * 100
                ead = x['NO_GOC'].sum()
                return pd.Series({'Fwd': fw, 'Rec': rc, 'Net': fw - rc, 'Total': t, 'EAD': ead})
            partner_roll = pair.groupby('DU_AN').apply(calc_partner).reset_index()

    roll_df = pd.DataFrame(results)

    # Aggregate Matrix
    agg_mat = pd.DataFrame()
    for mat in transitions.values():
        agg_mat = agg_mat.add(mat, fill_value=0) if not agg_mat.empty else mat
    if len(transitions) > 0:
        agg_mat = agg_mat / len(transitions)

    # ── 4. DPD Band Composition ──────────────────────────────────
    print("[4/6] Phân tích cơ cấu DPD theo tháng ...")
    band_comp = {}
    for m in MONTH_ORDER:
        if f'{m}_BAND' in pivot.columns:
            counts = pivot[f'{m}_BAND'].value_counts()
            band_comp[m] = counts
    band_comp_df = pd.DataFrame(band_comp).fillna(0)
    band_comp_df = band_comp_df.loc[[b[2] for b in DPD_BANDS if b[2] in band_comp_df.index]]
    band_comp_pct = band_comp_df.div(band_comp_df.sum()) * 100

    # ── 4B. NEW: Net Roll Rate per Band ─────────────────────────
    print("[4B/6] Net Roll Rate theo từng DPD Bucket (NEW)...")
    band_roll_stats = []
    if month_pairs:
        m1, m2 = month_pairs[-1]
        pair2 = pivot[[f'{m1}_BAND', f'{m2}_BAND', 'NO_GOC']].dropna(subset=[f'{m1}_BAND', f'{m2}_BAND'])
        pair2.columns = ['FROM_BAND', 'TO_BAND', 'NO_GOC']
        pair2 = pair2[(pair2['FROM_BAND'] != 'Unknown') & (pair2['TO_BAND'] != 'Unknown')]
        pair2['FROM_IDX'] = pair2['FROM_BAND'].map(band_order)
        pair2['TO_IDX']   = pair2['TO_BAND'].map(band_order)

        for band_label in all_bands:
            sub = pair2[pair2['FROM_BAND'] == band_label]
            if len(sub) < 10: continue
            fw  = (sub['TO_IDX'] > sub['FROM_IDX']).sum() / len(sub) * 100
            rc  = (sub['TO_IDX'] < sub['FROM_IDX']).sum() / len(sub) * 100
            st  = (sub['TO_IDX'] == sub['FROM_IDX']).sum() / len(sub) * 100
            net = fw - rc
            ead_fw  = sub.loc[sub['TO_IDX'] > sub['FROM_IDX'], 'NO_GOC'].sum()
            ead_tot = sub['NO_GOC'].sum()
            band_roll_stats.append({
                'Bucket':        band_label,
                'Tổng_HS':       len(sub),
                'Forward_%':     round(fw, 1),
                'Recovery_%':    round(rc, 1),
                'Stable_%':      round(st, 1),
                'Net_Roll_%':    round(net, 1),
                'EAD_Fwd_%':     round(ead_fw / ead_tot * 100, 1) if ead_tot > 0 else 0,
            })
    band_roll_df = pd.DataFrame(band_roll_stats)
    if not band_roll_df.empty:
        print("\n   Net Roll Rate theo DPD Bucket (Kỳ gần nhất):")
        print(band_roll_df[['Bucket','Tổng_HS','Forward_%','Recovery_%','Net_Roll_%']].to_string(index=False))

    # ── 4C. NEW: Forward-looking projection ─────────────────────
    projected_net_roll = None
    if len(roll_df) >= 2:
        # Simple linear projection
        net_rolls = roll_df['Net_Roll_%'].values
        trend = net_rolls[-1] - net_rolls[0]  # Average delta per period
        projected_net_roll = net_rolls[-1] + trend / max(len(net_rolls)-1, 1)
        print(f"\n   📈 Dự báo Net Roll Rate kỳ tới: {projected_net_roll:.2f}%")

    # KPI extraction
    total_hs, fwd_roll, rec_roll, net_roll, cure_rate, ead_net_roll = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    if len(results) > 0:
        latest = results[-1]
        total_hs    = latest['Tổng_HS']
        fwd_roll    = latest['Forward_Roll_%']
        rec_roll    = latest['Recovery_Roll_%']
        net_roll    = latest['Net_Roll_%']
        cure_rate   = latest['Cure_Rate_%']
        ead_net_roll = latest.get('EAD_Net_Roll_%', net_roll)

    insights_html = ""
    if net_roll > 5:
        trend_txt = f"⚠ DANH MỤC ĐANG XẤU ĐI (Net Roll = {net_roll:.1f}%)"
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>🔴 Báo động khẩn (Net Roll Rate = {net_roll:.1f}%):</strong> Tỷ lệ hồ sơ chuyển sang nợ xấu cao hơn tỷ lệ thu hồi. EAD-weighted Net Roll Rate = {ead_net_roll:.1f}%. Cần kích hoạt chiến dịch push calls mạnh ngay.
        </div>"""
    elif net_roll > 0:
        trend_txt = f"🟡 TÍCH LUỸ NỢ XẤU CHẬM (Net Roll = {net_roll:.1f}%)"
        insights_html += f"""
        <div class="alert-box alert-warn">
            <strong>🟡 Dấu hiệu rủi ro chớm nở:</strong> Net Roll Rate dương ({net_roll:.1f}%). EAD-Weighted: {ead_net_roll:.1f}%. Nợ đang già đi nhanh hơn tốc độ thu hồi.
        </div>"""
    else:
        trend_txt = f"🟢 DANH MỤC ĐANG PHỤC HỒI TỐT (Net Roll = {net_roll:.1f}%)"
        insights_html += f"""
        <div class="alert-box alert-success">
            <strong>🟢 Tốc độ thu hồi vượt kỳ vọng:</strong> Net Roll Rate âm ({net_roll:.1f}%). EAD-Weighted: {ead_net_roll:.1f}%. Chiến lược thu hồi đang phát huy tác dụng.
        </div>"""

    if cure_rate < 5:
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>⚠️ Cure Rate thấp ({cure_rate:.1f}%):</strong> Chỉ {cure_rate:.1f}% hồ sơ khó đòi có thể quay về trạng thái an toàn. Nợ sâu đang bế tắc.
        </div>"""

    if projected_net_roll is not None:
        color = "alert-danger" if projected_net_roll > 5 else ("alert-warn" if projected_net_roll > 0 else "alert-success")
        insights_html += f"""
        <div class="alert-box {color}">
            <strong>📈 Dự báo xu hướng tháng tới:</strong> Nếu tốc độ hiện tại duy trì, Net Roll Rate kỳ tới dự kiến khoảng <b>{projected_net_roll:.2f}%</b>. {'Cần can thiệp ngay để ngăn chặn vòng xoáy xấu.' if projected_net_roll > 3 else 'Xu hướng đang ổn định.'}
        </div>"""

    # ── 5. Visualization ─────────────────────────────────────────
    print("\n[5/6] Tạo Dashboard HTML cao cấp v3.0...")
    band_labels = [b[2] for b in DPD_BANDS]
    import plotly.offline as plo

    # Fig 1: Roll Rate Trend
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    cats  = roll_df['Cặp_Tháng']
    fig1.add_trace(go.Bar(name='Xấu Hơn (Fwd %)', x=cats, y=roll_df['Forward_Roll_%'], marker_color='#E53935'), secondary_y=False)
    fig1.add_trace(go.Bar(name='Tốt Hơn (Rec %)', x=cats, y=roll_df['Recovery_Roll_%'], marker_color='#43A047'), secondary_y=False)
    fig1.add_trace(go.Scatter(name='Net Roll Rate', x=cats, y=roll_df['Net_Roll_%'], mode='lines+markers', line=dict(color='#FDD835', width=3), marker=dict(size=9)), secondary_y=True)
    fig1.add_trace(go.Scatter(name='EAD Net Roll %', x=cats, y=roll_df['EAD_Net_Roll_%'], mode='lines+markers', line=dict(color='#FB8C00', width=2, dash='dot'), marker=dict(size=7)), secondary_y=True)
    fig1.update_layout(title="Xu Hướng Roll Rate & EAD-Weighted Net Roll", barmode='group', height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Fig 2: Heatmaps
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Ma Trận Aggregate (Trung Bình)", "Ma Trận Tháng Gần Nhất"), horizontal_spacing=0.1)
    if not agg_mat.empty:
        common = [b for b in band_labels if b in agg_mat.index and b in agg_mat.columns]
        z_vals = agg_mat.loc[common, common].values.round(1)
        fig2.add_trace(go.Heatmap(z=z_vals, x=common, y=common, colorscale='RdYlGn', text=z_vals.round(1), texttemplate="%{text}%", showscale=False), row=1, col=1)
    if month_pairs:
        latest_pair_key = f"{month_pairs[-1][0]}→{month_pairs[-1][1]}"
        mat = transitions[latest_pair_key]
        common = [b for b in band_labels if b in mat.index and b in mat.columns]
        z_vals = mat.loc[common, common].values.round(1)
        fig2.add_trace(go.Heatmap(z=z_vals, x=common, y=common, colorscale='RdYlGn', text=z_vals.round(1), texttemplate="%{text}%", showscale=True), row=1, col=2)
    fig2.update_layout(height=520, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=80, l=40, r=40))
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # Fig 3: Partner Roll Rate
    fig3 = go.Figure()
    if 'partner_roll' in locals() and not partner_roll.empty:
        p_sorted = partner_roll.sort_values('Net', ascending=False)
        fig3.add_trace(go.Bar(name='Net Roll %', x=p_sorted['DU_AN'], y=p_sorted['Net'], marker_color='#F59E0B'))
        fig3.add_trace(go.Bar(name='Forward %',  x=p_sorted['DU_AN'], y=p_sorted['Fwd'], marker_color='#E53935'))
        fig3.add_trace(go.Bar(name='Recovery %', x=p_sorted['DU_AN'], y=p_sorted['Rec'], marker_color='#43A047'))
        fig3.update_layout(title="Hiệu suất Thu Hồi Theo Đối Tác (Kỳ Gần Nhất)", barmode='group', height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 4: Net Roll per Band ───────────────────────────
    fig4 = go.Figure()
    if not band_roll_df.empty:
        colors_net = ['#E53935' if v > 5 else ('#FB8C00' if v > 0 else '#43A047') for v in band_roll_df['Net_Roll_%']]
        fig4.add_trace(go.Bar(
            y=band_roll_df['Bucket'], x=band_roll_df['Net_Roll_%'],
            orientation='h', marker_color=colors_net,
            text=[f"{v:+.1f}%" for v in band_roll_df['Net_Roll_%']], textposition='outside',
            hovertemplate="<b>%{y}</b><br>Net Roll: %{x:.1f}%<extra></extra>"
        ))
        fig4.add_vline(x=0, line_dash="dash", line_color="#FDD835")
        fig4.update_layout(title="Net Roll Rate Theo DPD Bucket (Kỳ Gần Nhất)", height=400,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=50, b=40, l=180, r=60), xaxis_title="Net Roll Rate (%)")
    div4 = plo.plot(fig4, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 5: DPD Band Composition Stacked ────────────────
    fig5 = go.Figure()
    palette = px.colors.qualitative.Bold
    for i, band in enumerate(band_comp_pct.index):
        fig5.add_trace(go.Bar(
            name=band, x=band_comp_pct.columns.tolist(), y=band_comp_pct.loc[band].values,
            marker_color=palette[i % len(palette)],
            hovertemplate=f"<b>{band}</b><br>Tháng: %{{x}}<br>Tỷ lệ: %{{y:.1f}}%<extra></extra>"
        ))
    fig5.update_layout(barmode='stack', title="Cơ Cấu DPD Bucket Theo Tháng (%)", height=450,
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       margin=dict(t=50, b=50, l=40, r=40))
    div5 = plo.plot(fig5, output_type='div', include_plotlyjs=False)

    # ─── Data Tables ────────────────────────────────────────────
    roll_summary_rows = ""
    for _, r in roll_df.iterrows():
        net_color = "#E53935" if r['Net_Roll_%'] > 5 else ("#FB8C00" if r['Net_Roll_%'] > 0 else "#10B981")
        roll_summary_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['Cặp_Tháng']}</td>
            <td style="text-align:right;">{r['Tổng_HS']:,}</td>
            <td style="text-align:right; color:#E53935;">{r['Forward_Roll_%']:.1f}%</td>
            <td style="text-align:right; color:#10B981;">{r['Recovery_Roll_%']:.1f}%</td>
            <td style="text-align:right;">{r['Stable_%']:.1f}%</td>
            <td style="text-align:right; font-weight:700; color:{net_color};">{r['Net_Roll_%']:+.2f}%</td>
            <td style="text-align:right;">{r.get('EAD_Net_Roll_%', 0):+.2f}%</td>
            <td style="text-align:right;">{r['Cure_Rate_%']:.1f}%</td>
        </tr>"""

    band_roll_rows = ""
    for _, r in band_roll_df.iterrows():
        net_color = "#E53935" if r['Net_Roll_%'] > 5 else ("#FB8C00" if r['Net_Roll_%'] > 0 else "#10B981")
        band_roll_rows += f"""
        <tr>
            <td style="font-weight:600; font-size:11.5px;">{r['Bucket']}</td>
            <td style="text-align:right;">{r['Tổng_HS']:,}</td>
            <td style="text-align:right; color:#E53935;">{r['Forward_%']:.1f}%</td>
            <td style="text-align:right; color:#10B981;">{r['Recovery_%']:.1f}%</td>
            <td style="text-align:right;">{r['Stable_%']:.1f}%</td>
            <td style="text-align:right; font-weight:700; color:{net_color};">{r['Net_Roll_%']:+.1f}%</td>
            <td style="text-align:right;">{r['EAD_Fwd_%']:.1f}%</td>
        </tr>"""

    proj_txt = f"{projected_net_roll:.2f}%" if projected_net_roll is not None else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>7E — Roll Rate Matrix v3.0</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg: #0F172A; --card: #1E293B; --primary: #38BDF8;
    --success: #10B981; --danger: #EF4444; --warn: #F59E0B;
    --muted: #64748B; --border: #334155; --text: #F1F5F9; --radius: 12px;
}}
body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); padding: 24px; margin: 0; }}
.header {{ background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; margin-bottom: 24px; }}
.header h1 {{ margin: 0 0 8px 0; font-size: 22px; font-weight: 800; color: var(--primary); }}
.header p {{ margin: 0; font-size: 13px; color: var(--muted); }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.kpi-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; position: relative; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); }}
.kpi-card::after {{ content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: var(--primary); border-radius: 0 0 var(--radius) var(--radius); }}
.kpi-card.kpi-success::after {{ background: var(--success); }}
.kpi-card.kpi-danger::after {{ background: var(--danger); }}
.kpi-card.kpi-warn::after {{ background: var(--warn); }}
.kpi-label {{ font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 22px; font-weight: 800; line-height: 1.1; }}
.kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); margin-bottom: 20px; }}
.alert-trend {{ background: #1E293B; border-left: 4px solid var(--primary); padding: 14px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 13px; border: 1px solid var(--border); border-left-width: 4px; }}
.tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; flex-wrap: wrap; }}
.tab-btn {{ background: transparent; color: var(--muted); border: none; padding: 10px 16px; font-size: 13px; font-weight: 600; cursor: pointer; border-radius: 8px; transition: all 0.2s; }}
.tab-btn:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
.tab-btn.active {{ background: var(--primary); color: #000; }}
.tab-content {{ display: none; animation: fadeIn 0.3s ease; }}
.tab-content.active {{ display: block; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.alert-box {{ padding: 14px 16px; border-radius: 8px; margin-bottom: 14px; font-size: 13.5px; border-left: 4px solid; }}
.alert-danger {{ background: rgba(239,68,68,0.1); border-left-color: var(--danger); color: #FCA595; }}
.alert-warn {{ background: rgba(245,158,11,0.1); border-left-color: var(--warn); color: #FCD34D; }}
.alert-success {{ background: rgba(16,185,129,0.1); border-left-color: var(--success); color: #6EE7B7; }}
.data-table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
.data-table th {{ background: #0F172A; padding: 10px 14px; font-weight: 700; color: var(--primary); border-bottom: 2px solid var(--border); text-align: left; white-space: nowrap; }}
.data-table td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
.data-table tr:hover {{ background: rgba(56,189,248,0.05); }}
        :root:not([data-theme="dark"]) {{
            --bg: #F8FAFC !important; --card: #FFFFFF !important; --text: #0F172A !important;
            --border: #E2E8F0 !important; --muted: #64748B !important; --primary: #2563EB !important;
            --success: #10B981 !important; --danger: #EF4444 !important;
            --warning: #F59E0B !important; --warn: #F59E0B !important; --info: #06B6D4 !important;
        }}
        html:not([data-theme="dark"]) body {{ background-color: var(--bg) !important; color: var(--text) !important; }}
        html:not([data-theme="dark"]) .kpi-card, html:not([data-theme="dark"]) .chart-card, html:not([data-theme="dark"]) .header {{ background-color: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; }}
        html:not([data-theme="dark"]) h1, html:not([data-theme="dark"]) h2, html:not([data-theme="dark"]) h3 {{ color: var(--text) !important; }}
        html:not([data-theme="dark"]) .kpi-label, html:not([data-theme="dark"]) .kpi-sub {{ color: var(--muted) !important; }}
        html:not([data-theme="dark"]) .data-table th {{ background-color: rgba(37,99,235,0.05) !important; color: #1E3A8A !important; }}
        html:not([data-theme="dark"]) .data-table td {{ color: var(--text) !important; border-color: var(--border) !important; }}
        html:not([data-theme="dark"]) .tab-btn.active {{ color: var(--primary) !important; background-color: rgba(37,99,235,0.08) !important; }}
</style>
</head>
<body>
<div class="header">
    <h1>7E — ROLL RATE MATRIX <span style="font-size:14px; color:#64748B;">v3.0 | Risk Manager Edition</span></h1>
    <p>Phân tích động thái danh mục: Roll Rate, EAD-Weighted Net Roll, Net Roll Rate per DPD Bucket, Dự báo xu hướng — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Hồ Sơ Phân Tích</div>
        <div class="kpi-value">{total_hs:,} HS</div>
        <div class="kpi-sub">Số hồ sơ có lịch sử liên tục</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Forward Roll Rate</div>
        <div class="kpi-value">{fwd_roll:.1f}%</div>
        <div class="kpi-sub">Tỷ lệ hồ sơ xấu hơn</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Recovery Roll Rate</div>
        <div class="kpi-value">{rec_roll:.1f}%</div>
        <div class="kpi-sub">Tỷ lệ hồ sơ cải thiện</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Net Roll Rate</div>
        <div class="kpi-value">{net_roll:+.2f}%</div>
        <div class="kpi-sub">Gia tốc nợ xấu ròng</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">EAD-Weighted Net Roll</div>
        <div class="kpi-value">{ead_net_roll:+.2f}%</div>
        <div class="kpi-sub">Tính theo giá trị nợ gốc</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Cure Rate</div>
        <div class="kpi-value">{cure_rate:.1f}%</div>
        <div class="kpi-sub">Từ Nợ Khó Đòi → An Toàn</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Dự Báo Kỳ Tới</div>
        <div class="kpi-value">{proj_txt}</div>
        <div class="kpi-sub">Net Roll Rate dự phóng tuyến tính</div>
    </div>
</div>
<div class="alert-trend">
    📢 <strong>Đánh giá xu hướng:</strong> {trend_txt}
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event,'tab1')">1. Xu Hướng Roll Rate</button>
    <button class="tab-btn" onclick="openTab(event,'tab2')">2. Roll Rate per Bucket 🆕</button>
    <button class="tab-btn" onclick="openTab(event,'tab3')">3. Heatmap Chuyển Dịch</button>
    <button class="tab-btn" onclick="openTab(event,'tab4')">4. Cơ Cấu DPD Stacked 🆕</button>
    <button class="tab-btn" onclick="openTab(event,'tab5')">5. Roll Rate Đối Tác</button>
    <button class="tab-btn" onclick="openTab(event,'tab6')">6. Bảng Dữ Liệu 🆕</button>
    <button class="tab-btn" onclick="openTab(event,'tab7')">7. Insights & Chiến Lược</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="chart-card">{div1}</div>
</div>
<div id="tab2" class="tab-content">
    <div class="chart-card">{div4}</div>
    <div class="chart-card">
        <h3 style="margin:0 0 14px 0; color:var(--primary); font-size:15px;">📊 Net Roll Rate Theo DPD Bucket (Kỳ Gần Nhất)</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>DPD Bucket</th><th>Số HS</th><th>Forward %</th>
                <th>Recovery %</th><th>Stable %</th><th>Net Roll %</th><th>EAD Forward %</th>
            </tr></thead>
            <tbody>{band_roll_rows}</tbody>
        </table></div>
    </div>
</div>
<div id="tab3" class="tab-content">
    <div class="chart-card">{div2}</div>
</div>
<div id="tab4" class="tab-content">
    <div class="chart-card">{div5}</div>
</div>
<div id="tab5" class="tab-content">
    <div class="chart-card">{div3}</div>
</div>
<div id="tab6" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 14px 0; color:var(--primary); font-size:15px;">📋 Bảng Tóm Tắt Roll Rate Theo Kỳ</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Kỳ</th><th>Tổng HS</th><th>Forward %</th><th>Recovery %</th>
                <th>Stable %</th><th>Net Roll %</th><th>EAD Net Roll %</th><th>Cure Rate %</th>
            </tr></thead>
            <tbody>{roll_summary_rows}</tbody>
        </table></div>
    </div>
</div>
<div id="tab7" class="tab-content">
    <div class="chart-card" style="padding:28px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom:20px;">💡 Chiến Lược Can Thiệp Dựa Trên Dữ Liệu</h3>
        {insights_html}
    </div>
</div>

<script>
function openTab(evt, tabName) {{
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
    window.dispatchEvent(new Event('resize'));
}}
(function() {{
    function updatePlotlyTheme(isDark) {{
        const textCol = isDark ? '#F1F5F9' : '#0F172A';
        const gridCol = isDark ? '#334155' : '#E2E8F0';
        document.querySelectorAll('.plotly-graph-div').forEach(div => {{
            try {{ Plotly.relayout(div, {{ 'paper_bgcolor':'rgba(0,0,0,0)', 'plot_bgcolor':'rgba(0,0,0,0)', 'font.color':textCol, 'xaxis.gridcolor':gridCol, 'yaxis.gridcolor':gridCol }}); }} catch(e) {{}}
        }});
    }}
    const theme = localStorage.getItem('vne_theme');
    if (theme === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
    window.addEventListener('message', function(e) {{
        if (e.data && typeof e.data.theme === 'string') {{
            const isDark = e.data.theme === 'dark';
            isDark ? document.documentElement.setAttribute('data-theme','dark') : document.documentElement.removeAttribute('data-theme');
            setTimeout(() => {{ updatePlotlyTheme(isDark); window.dispatchEvent(new Event('resize')); }}, 50);
        }}
    }});
    try {{ window.parent.postMessage({{ type: 'request_theme' }}, '*'); }} catch(e) {{}}
    window.addEventListener('DOMContentLoaded', () => {{
        setTimeout(() => {{ updatePlotlyTheme(document.documentElement.getAttribute('data-theme') === 'dark'); }}, 500);
    }});
}})();
</script>
</body>
</html>"""

    # ── 6. Output ─────────────────────────────────────────────────
    print("\n[6/6] Lưu output...")
    out_html = os.path.join(REPORT_DIR, "7e_ROLL_RATE.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv = os.path.join(SUB_DATA_DIR, "7e_roll_rate.csv")
    roll_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    if month_pairs:
        latest_pair = f"{month_pairs[-1][0]}→{month_pairs[-1][1]}"
        if latest_pair in transitions:
            latest_matrix = transitions[latest_pair]
            latest_matrix = latest_matrix.reindex(index=band_labels, columns=band_labels, fill_value=0.0)
            out_matrix_csv = os.path.join(SUB_DATA_DIR, "7e_roll_rate_matrix.csv")
            latest_matrix.to_csv(out_matrix_csv, encoding='utf-8-sig')
            print(f"   → Matrix: {out_matrix_csv}")

    print(f"\n✅ HOÀN THÀNH MODULE 7E v3.0!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")
    if projected_net_roll is not None:
        print(f"\n📌 KEY INSIGHT: Net Roll Rate hiện tại = {net_roll:.2f}% | Dự báo kỳ tới = {projected_net_roll:.2f}%")

if __name__ == "__main__":
    run()
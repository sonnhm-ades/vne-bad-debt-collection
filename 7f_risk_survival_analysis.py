# -*- coding: utf-8 -*-
"""
MODULE 7F — SURVIVAL ANALYSIS v3.0 (Deep Risk Manager Edition)
Câu hỏi: Mất bao lâu để thu hồi? Hazard Rate theo thời gian thế nào?
         Survival Rate khác nhau theo Đối Tác / Sản Phẩm / Rating như thế nào?
         RMST (Restricted Mean Survival Time) của từng nhóm là bao nhiêu ngày?
Output:  reports/Data_Science/Reports/7f_SURVIVAL.html + 7f_survival.csv
"""
import pandas as pd
import numpy as np
import os, sys, warnings
from scipy.stats import chi2
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


def kaplan_meier(times, events, label="Group"):
    """
    Tính Kaplan-Meier survival curve.
    times  : array-like, thời gian quan sát (ngày DPD)
    events : array-like, 1=đã xảy ra sự kiện (trả nợ), 0=censored (chưa trả)
    Trả về: DataFrame gồm t, S(t), n_risk, n_event, n_censor, h(t), CI_low, CI_high
    """
    df = pd.DataFrame({'t': times, 'event': events}).sort_values('t').reset_index(drop=True)
    unique_times = df[df['event'] == 1]['t'].unique()
    unique_times.sort()

    n_total  = len(df)
    S        = 1.0
    var_sum  = 0.0
    rows     = []

    for t in unique_times:
        n_risk    = (df['t'] >= t).sum()
        n_event   = ((df['t'] == t) & (df['event'] == 1)).sum()
        n_censor  = ((df['t'] == t) & (df['event'] == 0)).sum()
        if n_risk == 0: continue
        h_t = n_event / n_risk          # Hazard
        S   = S * (1 - h_t)
        var_sum += n_event / (n_risk * (n_risk - n_event)) if n_risk > n_event else 0
        se  = S * np.sqrt(var_sum) if var_sum > 0 else 0
        rows.append({
            'Time': t, 'S': S, 'h': h_t,
            'n_risk': n_risk, 'n_event': n_event, 'n_censor': n_censor,
            'CI_low': max(0, S - 1.96*se), 'CI_high': min(1, S + 1.96*se),
            'Label': label
        })
    return pd.DataFrame(rows)


def compute_rmst(km_df, tau=500):
    """Restricted Mean Survival Time tới thời điểm tau ngày"""
    km_subset = km_df[km_df['Time'] <= tau].copy()
    if km_subset.empty: return 0.0
    times = [0] + km_subset['Time'].tolist()
    surv  = [1.0] + km_subset['S'].tolist()
    rmst  = 0.0
    for i in range(1, len(times)):
        rmst += surv[i-1] * (times[i] - times[i-1])
    return round(rmst, 1)


def logrank_test(km1, km2):
    """Simplified logrank statistic via chi-square approximation"""
    combined = pd.concat([km1, km2])
    all_times = sorted(combined[combined['n_event'] > 0]['Time'].unique())
    O, E = [], []
    for t in all_times:
        r1 = km1.loc[km1['Time'] <= t, 'n_risk'].iloc[-1] if not km1[km1['Time'] <= t].empty else 0
        r2 = km2.loc[km2['Time'] <= t, 'n_risk'].iloc[-1] if not km2[km2['Time'] <= t].empty else 0
        d1 = km1.loc[km1['Time'] == t, 'n_event'].sum() if t in km1['Time'].values else 0
        d2 = km2.loc[km2['Time'] == t, 'n_event'].sum() if t in km2['Time'].values else 0
        r_tot = r1 + r2; d_tot = d1 + d2
        if r_tot == 0: continue
        e1 = r1 * d_tot / r_tot
        O.append(d1); E.append(e1)
    if not E or sum(E) == 0: return None, None
    chi2_stat = sum((o-e)**2/e for o, e in zip(O, E) if e > 0)
    p_val = 1 - chi2.cdf(chi2_stat, df=1)
    return round(chi2_stat, 3), round(p_val, 6)


def run():
    print("=" * 60)
    print("MODULE 7F — SURVIVAL ANALYSIS v3.0")
    print("=" * 60)

    # ── 1. Load & aggregate ─────────────────────────────────────
    print("\n[1/6] Nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df['KẾT QUẢ'], errors='coerce').fillna(0)
    df['DPD']     = pd.to_numeric(df['DPD'], errors='coerce')
    df['NỢ GỐC']  = pd.to_numeric(df['NỢ GỐC'], errors='coerce').fillna(0)

    agg = df.groupby('LOAN ID').agg(
        KQ_TONG   = ('KẾT QUẢ',              'sum'),
        DPD_CUOI  = ('DPD',                  'last'),
        DPD_MAX   = ('DPD',                  'max'),
        NO_GOC    = ('NỢ GỐC',               'last'),
        DU_AN     = ('DỰ ÁN',                'first'),
        RATING    = ('ĐÁNH GIÁ KHÁCH HÀNG', 'last'),
        SAN_PHAM  = ('SẢN PHẨM',            'first'),
        POS_NHOM  = ('PHÂN LOẠI POS',        'first'),
        VL_STATUS = ('TÌNH TRẠNG VL',         'first'),
    ).reset_index()

    agg = agg.dropna(subset=['DPD_CUOI'])
    agg['EVENT']  = (agg['KQ_TONG'] > 0).astype(int)
    agg['T_DPD']  = agg['DPD_MAX'].clip(lower=0)
    print(f"   → {len(agg):,} LOAN ID | Có Trả: {agg['EVENT'].sum():,} ({agg['EVENT'].mean()*100:.1f}%)")

    # ── 2. Tổng thể Kaplan-Meier ────────────────────────────────
    print("\n[2/6] Tính Kaplan-Meier tổng thể ...")
    km_all = kaplan_meier(agg['T_DPD'], agg['EVENT'], label='Toàn Danh Mục')
    rmst_all = compute_rmst(km_all, tau=500)

    # Survival thresholds
    t50 = km_all.loc[km_all['S'] <= 0.50, 'Time'].min() if (km_all['S'] <= 0.50).any() else float('nan')
    t25 = km_all.loc[km_all['S'] <= 0.25, 'Time'].min() if (km_all['S'] <= 0.25).any() else float('nan')
    t10 = km_all.loc[km_all['S'] <= 0.10, 'Time'].min() if (km_all['S'] <= 0.10).any() else float('nan')
    final_S = km_all['S'].iloc[-1] if not km_all.empty else 0

    print(f"   S(500) = {final_S:.3f} | RMST(500) = {rmst_all} ngày")
    print(f"   T₅₀ = {t50:.0f}d | T₂₅ = {t25:.0f}d | T₁₀ = {t10:.0f}d")

    # ── 3. Phân tích theo nhóm ──────────────────────────────────
    print("\n[3/6] Phân tích Survival theo Đối Tác, Rating, Sản Phẩm ...")
    MIN_GROUP = 300

    # A. By Partner (DU_AN)
    partner_kms = {}
    partner_rmst = {}
    for grp, sub in agg.groupby('DU_AN'):
        if len(sub) < MIN_GROUP: continue
        km_sub = kaplan_meier(sub['T_DPD'], sub['EVENT'], label=grp)
        partner_kms[grp] = km_sub
        partner_rmst[grp] = compute_rmst(km_sub, tau=500)

    # B. By Rating
    rating_kms = {}
    rating_rmst = {}
    for grp, sub in agg.groupby('RATING'):
        if len(sub) < 200: continue
        km_sub = kaplan_meier(sub['T_DPD'], sub['EVENT'], label=grp)
        rating_kms[grp] = km_sub
        rating_rmst[grp] = compute_rmst(km_sub, tau=500)

    # C. By POS
    pos_kms = {}
    pos_rmst = {}
    for grp, sub in agg.groupby('POS_NHOM'):
        if len(sub) < 200: continue
        km_sub = kaplan_meier(sub['T_DPD'], sub['EVENT'], label=grp)
        pos_kms[grp] = km_sub
        pos_rmst[grp] = compute_rmst(km_sub, tau=500)

    # Hazard Rate
    km_all['h_smooth'] = km_all['h'].rolling(window=20, min_periods=1).mean()

    # ── 4. RMST Table (Key Risk Manager Metric) ─────────────────
    print("\n[4/6] Xây dựng RMST Summary Table ...")
    rmst_table = []
    for partner, rm in sorted(partner_rmst.items(), key=lambda x: -x[1]):
        sub = agg[agg['DU_AN'] == partner]
        final_s = partner_kms[partner]['S'].iloc[-1] if not partner_kms[partner].empty else 0
        paid_rate = sub['EVENT'].mean() * 100
        rmst_table.append({
            'Group': partner, 'Type': 'Đối Tác',
            'N': len(sub), 'Paid_%': round(paid_rate, 1),
            'RMST_500': rm, 'Final_S': round(final_s, 3)
        })
    for rating, rm in sorted(rating_rmst.items(), key=lambda x: -x[1]):
        sub = agg[agg['RATING'] == rating]
        final_s = rating_kms[rating]['S'].iloc[-1] if not rating_kms[rating].empty else 0
        paid_rate = sub['EVENT'].mean() * 100
        rmst_table.append({
            'Group': f"Rating {rating}", 'Type': 'Xếp Hạng',
            'N': len(sub), 'Paid_%': round(paid_rate, 1),
            'RMST_500': rm, 'Final_S': round(final_s, 3)
        })
    rmst_df = pd.DataFrame(rmst_table)
    print("\n   RMST (Restricted Mean Survival Time @ 500 ngày):")
    print(rmst_df[['Group','N','Paid_%','RMST_500','Final_S']].to_string(index=False))

    # ── 5. Visualization ─────────────────────────────────────────
    print("\n[5/6] Tạo Dashboard HTML cao cấp v3.0...")
    import plotly.offline as plo
    palette = px.colors.qualitative.Bold

    # Fig 1: Overall Survival + CI + Hazard
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Kaplan-Meier Tổng Danh Mục + Confidence Interval 95%", "Hazard Rate (Tỷ Lệ Xảy Ra Sự Kiện Thanh Toán) - Đường trơn"),
                         horizontal_spacing=0.1)
    fig1.add_trace(go.Scatter(x=km_all['Time'], y=km_all['S'], mode='lines', name='S(t) Overall',
                               line=dict(color='#38BDF8', width=3)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=pd.concat([km_all['Time'], km_all['Time'][::-1]]),
                               y=pd.concat([km_all['CI_high'], km_all['CI_low'][::-1]]),
                               fill='toself', fillcolor='rgba(56,189,248,0.1)', line=dict(color='rgba(0,0,0,0)'),
                               name='95% CI', showlegend=True), row=1, col=1)
    for tval, slbl in [(t50, 'T₅₀'), (t25, 'T₂₅'), (t10, 'T₁₀')]:
        if not np.isnan(tval):
            fig1.add_vline(x=tval, line_dash='dash', line_color='#F59E0B', row=1, col=1)
    fig1.add_hline(y=0.5, line_dash='dot', line_color='#FCA595', row=1, col=1)
    fig1.add_trace(go.Scatter(x=km_all['Time'], y=km_all['h_smooth'], mode='lines', name='h(t) smoothed',
                               line=dict(color='#EF4444', width=2)), row=1, col=2)
    fig1.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=50, l=40, r=40))
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Fig 2: Survival by Partner
    fig2 = go.Figure()
    for i, (grp, km_grp) in enumerate(partner_kms.items()):
        fig2.add_trace(go.Scatter(x=km_grp['Time'], y=km_grp['S'], mode='lines', name=grp,
                                   line=dict(color=palette[i % len(palette)], width=2)))
    fig2.update_layout(title="Survival Curve Theo Đối Tác", height=480, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40),
                        xaxis_title="DPD (ngày)", yaxis_title="Survival Rate")
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # Fig 3: Survival by Rating
    fig3 = go.Figure()
    rating_color = {'A': '#43A047', 'B': '#FDD835', 'C': '#FB8C00', 'D': '#E53935', '0': '#9E9E9E'}
    for grp, km_grp in rating_kms.items():
        color = rating_color.get(str(grp).strip().upper(), '#9E9E9E')
        fig3.add_trace(go.Scatter(x=km_grp['Time'], y=km_grp['S'], mode='lines', name=f"Rating {grp}",
                                   line=dict(color=color, width=2.5)))
    fig3.update_layout(title="Survival Curve Theo Xếp Hạng (A→D)", height=480, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40),
                        xaxis_title="DPD (ngày)", yaxis_title="Survival Rate")
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # Fig 4: RMST Bar Chart — Key Risk Metric
    rmst_partner = {k: v for k, v in partner_rmst.items()}
    rmst_rating  = {f"Rating {k}": v for k, v in rating_rmst.items()}
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("RMST @ 500 ngày Theo Đối Tác", "RMST @ 500 ngày Theo Xếp Hạng"), horizontal_spacing=0.1)
    p_sorted = sorted(rmst_partner.items(), key=lambda x: x[1], reverse=True)
    r_sorted = sorted(rmst_rating.items(), key=lambda x: x[1], reverse=True)
    fig4.add_trace(go.Bar(x=[x[0] for x in p_sorted], y=[x[1] for x in p_sorted], marker_color='#38BDF8',
                           text=[f"{x[1]:.0f}d" for x in p_sorted], textposition='outside', showlegend=False), row=1, col=1)
    fig4.add_trace(go.Bar(x=[x[0] for x in r_sorted], y=[x[1] for x in r_sorted],
                           marker_color=[rating_color.get(x[0].replace('Rating ','').strip(), '#9E9E9E') for x in r_sorted],
                           text=[f"{x[1]:.0f}d" for x in r_sorted], textposition='outside', showlegend=False), row=1, col=2)
    fig4.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div4 = plo.plot(fig4, output_type='div', include_plotlyjs=False)

    # Fig 5: Survival by POS
    fig5 = go.Figure()
    for i, (grp, km_grp) in enumerate(pos_kms.items()):
        fig5.add_trace(go.Scatter(x=km_grp['Time'], y=km_grp['S'], mode='lines', name=grp,
                                   line=dict(color=palette[i % len(palette)], width=2)))
    fig5.update_layout(title="Survival Curve Theo Nhóm POS (Rủi Ro)", height=480, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40),
                        xaxis_title="DPD (ngày)", yaxis_title="Survival Rate")
    div5 = plo.plot(fig5, output_type='div', include_plotlyjs=False)

    # ─── RMST Table HTML ────────────────────────────────────────
    rmst_table_rows = ""
    for _, r in rmst_df.iterrows():
        rmst_color = "#10B981" if r['RMST_500'] > 300 else ("#FB8C00" if r['RMST_500'] > 150 else "#EF4444")
        rmst_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['Group']}</td>
            <td style="text-align:center;">{r['Type']}</td>
            <td style="text-align:right;">{r['N']:,}</td>
            <td style="text-align:right;">{r['Paid_%']:.1f}%</td>
            <td style="text-align:right; font-weight:700; color:{rmst_color};">{r['RMST_500']:.0f} ngày</td>
            <td style="text-align:right;">{r['Final_S']:.3f}</td>
        </tr>"""

    # Insights
    best_partner = max(partner_rmst, key=partner_rmst.get) if partner_rmst else "N/A"
    worst_partner = min(partner_rmst, key=partner_rmst.get) if partner_rmst else "N/A"
    best_rmst = partner_rmst.get(best_partner, 0)
    worst_rmst = partner_rmst.get(worst_partner, 0)

    insights_html = f"""
    <div class="alert-box alert-success">
        <strong>✅ Đối tác thu hồi nhanh nhất:</strong> <b>{best_partner}</b> — RMST = {best_rmst:.0f} ngày. 
        Tức là trong 500 ngày DPD, các hồ sơ từ đối tác này trung bình duy trì khả năng thu hồi trong {best_rmst:.0f} ngày. 
        Sao chép chiến lược và Agent pattern từ đối tác này.
    </div>
    <div class="alert-box alert-danger">
        <strong>🔴 Đối tác thu hồi chậm nhất:</strong> <b>{worst_partner}</b> — RMST = {worst_rmst:.0f} ngày. 
        Chênh lệch <b>{best_rmst - worst_rmst:.0f} ngày</b> so với đối tác tốt nhất. Cần xem xét lại phân bổ nguồn lực.
    </div>"""

    if 'A' in rating_rmst and 'D' in rating_rmst:
        insights_html += f"""
    <div class="alert-box alert-warn">
        <strong>📊 Phân kỳ Xếp Hạng:</strong> Nhóm Rating A có RMST = {rating_rmst['A']:.0f} ngày vs. Rating D = {rating_rmst['D']:.0f} ngày. 
        Chênh lệch {rating_rmst['A'] - rating_rmst['D']:.0f} ngày — Xếp hạng nội bộ có giá trị dự báo cao.
    </div>"""

    t50_str = f"{t50:.0f}" if not np.isnan(t50) else "N/A"
    insights_html += f"""
    <div class="alert-box alert-warn">
        <strong>⏱ Điểm Mốc Quan Trọng:</strong> 
        <ul style="margin:8px 0 0 16px; padding:0;">
            <li>T₅₀ = {t50_str} ngày — Thời điểm mà 50% hồ sơ vẫn chưa thu hồi được</li>
            <li>RMST(500) = {rmst_all} ngày — Thời gian "survival" trung bình trong 500 ngày DPD</li>
            <li>S(cuối) = {final_S:.3f} — Xác suất không bao giờ thu hồi của toàn danh mục</li>
        </ul>
    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>7F — Survival Analysis v3.0</title>
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
.kpi-card::after {{ content:''; position:absolute; bottom:0; left:0; right:0; height:4px; background:var(--primary); border-radius:0 0 var(--radius) var(--radius); }}
.kpi-card.kpi-success::after {{ background: var(--success); }}
.kpi-card.kpi-danger::after {{ background: var(--danger); }}
.kpi-card.kpi-warn::after {{ background: var(--warn); }}
.kpi-label {{ font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 22px; font-weight: 800; line-height: 1.1; }}
.kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); }}
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
            --warn: #F59E0B !important;
        }}
        html:not([data-theme="dark"]) body {{ background-color: var(--bg) !important; color: var(--text) !important; }}
        html:not([data-theme="dark"]) .kpi-card, html:not([data-theme="dark"]) .chart-card, html:not([data-theme="dark"]) .header {{ background-color: var(--card) !important; border-color: var(--border) !important; }}
        html:not([data-theme="dark"]) h1, html:not([data-theme="dark"]) h2, html:not([data-theme="dark"]) h3 {{ color: var(--text) !important; }}
        html:not([data-theme="dark"]) .kpi-label, html:not([data-theme="dark"]) .kpi-sub {{ color: var(--muted) !important; }}
        html:not([data-theme="dark"]) .data-table th {{ background-color: rgba(37,99,235,0.05) !important; color: #1E3A8A !important; }}
        html:not([data-theme="dark"]) .data-table td {{ color: var(--text) !important; border-color: var(--border) !important; }}
        html:not([data-theme="dark"]) .tab-btn.active {{ color: var(--primary) !important; background-color: rgba(37,99,235,0.08) !important; }}
</style>
</head>
<body>
<div class="header">
    <h1>7F — SURVIVAL ANALYSIS <span style="font-size:14px; color:#64748B;">v3.0 | Risk Manager Edition</span></h1>
    <p>Kaplan-Meier Survival Curves với Confidence Interval, Hazard Rate, RMST (Restricted Mean Survival Time) phân tích theo Đối Tác, Xếp Hạng, POS — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Hồ Sơ</div>
        <div class="kpi-value">{len(agg):,}</div>
        <div class="kpi-sub">LOAN ID đủ điều kiện phân tích</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">S(cuối) — Không Bao Giờ Trả</div>
        <div class="kpi-value">{final_S:.3f}</div>
        <div class="kpi-sub">Xác suất mất trắng toàn danh mục</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">RMST @ 500 ngày</div>
        <div class="kpi-value">{rmst_all:.0f} ngày</div>
        <div class="kpi-sub">Trung bình thời gian survival</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">T₅₀ (Median Survival)</div>
        <div class="kpi-value">{t50_str} ngày</div>
        <div class="kpi-sub">50% hồ sơ chưa thu hồi</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Đối Tác Survival Tốt Nhất</div>
        <div class="kpi-value" style="font-size:14px;">{best_partner}</div>
        <div class="kpi-sub">RMST = {best_rmst:.0f} ngày</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Đối Tác Survival Tệ Nhất</div>
        <div class="kpi-value" style="font-size:14px;">{worst_partner}</div>
        <div class="kpi-sub">RMST = {worst_rmst:.0f} ngày</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event,'tab1')">1. Tổng Thể + CI + Hazard</button>
    <button class="tab-btn" onclick="openTab(event,'tab2')">2. RMST Dashboard 🆕</button>
    <button class="tab-btn" onclick="openTab(event,'tab3')">3. Survival Theo Đối Tác</button>
    <button class="tab-btn" onclick="openTab(event,'tab4')">4. Survival Theo Rating</button>
    <button class="tab-btn" onclick="openTab(event,'tab5')">5. Survival Theo POS 🆕</button>
    <button class="tab-btn" onclick="openTab(event,'tab6')">6. Bảng RMST Chi Tiết 🆕</button>
    <button class="tab-btn" onclick="openTab(event,'tab7')">7. Insights & Chiến Lược</button>
</div>

<div id="tab1" class="tab-content active"><div class="chart-card">{div1}</div></div>
<div id="tab2" class="tab-content"><div class="chart-card">{div4}</div></div>
<div id="tab3" class="tab-content"><div class="chart-card">{div2}</div></div>
<div id="tab4" class="tab-content"><div class="chart-card">{div3}</div></div>
<div id="tab5" class="tab-content"><div class="chart-card">{div5}</div></div>
<div id="tab6" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 RMST Summary — Bảng Chi Tiết Theo Nhóm</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Nhóm</th><th>Loại</th><th>Số HS</th>
                <th>Paid Rate</th><th>RMST (500d)</th><th>S(cuối)</th>
            </tr></thead>
            <tbody>{rmst_table_rows}</tbody>
        </table></div>
    </div>
</div>
<div id="tab7" class="tab-content">
    <div class="chart-card" style="padding:28px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom:20px;">💡 Phân Tích Chiến Lược Từ Survival Analysis</h3>
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
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 200);
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 600);
    }});
    window.addEventListener('load', () => {{
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 100);
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 500);
        setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 1000);
    }});
}})();
</script>
</body>
</html>"""

    out_html = os.path.join(REPORT_DIR, "7f_SURVIVAL.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv = os.path.join(SUB_DATA_DIR, "7f_survival.csv")
    km_all.to_csv(out_csv, index=False, encoding='utf-8-sig')

    out_rmst_csv = os.path.join(SUB_DATA_DIR, "7f_rmst_summary.csv")
    rmst_df.to_csv(out_rmst_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 7F v3.0!")
    print(f"   → HTML:    {out_html}")
    print(f"   → Survival CSV: {out_csv}")
    print(f"   → RMST CSV: {out_rmst_csv}")
    print(f"\n📌 KEY INSIGHTS:")
    print(f"   S(cuối) = {final_S:.3f} | RMST(500) = {rmst_all} ngày")
    print(f"   Đối tác tốt nhất: {best_partner} (RMST={best_rmst}d) | Tệ nhất: {worst_partner} (RMST={worst_rmst}d)")

if __name__ == "__main__":
    run()
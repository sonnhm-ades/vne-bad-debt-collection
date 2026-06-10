# -*- coding: utf-8 -*-
"""
MODULE 8G — MASTER COMMAND CENTER DASHBOARD (2 TABS: OPERATIONAL + RISK)
Phiên bản: 1.0 (2026-05-26) — Merge 8G (Operational) + 9F (Risk)
Thiết kế: STRATEGY_HUB standard (Inter font, sidebar, dark/light mode, KPI cards, Plotly CDN)

Tab 1 — OPERATIONAL: ARR YTD, RPC Rate, Kept PTP, drill-down theo DỰ ÁN/Chi nhánh/CỤM_HÀNH_VI/DPD
Tab 2 — RISK:        LGD gauge, Roll-Rate heatmap, Survival proxy, Feedback Loop status

Baseline (100% từ data thực):
  - Nợ gốc tổng (EAD):   đọc từ CLEANED.csv   (không hard-code)
  - Kết quả YTD:          đọc từ CLEANED.csv
  - ARR:                  KẾT QUẢ / NỢ GỐC     (không hard-code)
  - ARR Feedback Threshold: 0.5% (lấy từ feedback_thresholds.json nếu có)

Output: reports/Data_Science/Reports/8g_MASTER_DASHBOARD.html
"""
import pandas as pd
import numpy as np
import os, sys, json, warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as plo

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026'
OUTPUT_DIR    = os.path.join(BASE, 'reports', 'Data_Science')
REPORT_DIR    = os.path.join(OUTPUT_DIR, 'Reports')
DATA_DIR      = os.path.join(OUTPUT_DIR, 'Data')
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

CLEANED_PATH  = os.path.join(BASE, 'TỔNG HỢP NĂM 2026 CLEANED.csv')
DS_SEG_PATH   = os.path.join(DATA_DIR, 'DS_SEGMENTATION_FINAL.csv')
FEEDBACK_PATH = os.path.join(DATA_DIR, 'feedback_thresholds.json')
LGD_CSV_PATH  = os.path.join(DATA_DIR, '7d_lgd_results.csv')
ROLLRATE_CSV  = os.path.join(DATA_DIR, '7e_roll_rate_matrix.csv')

# ── Design Tokens (STRATEGY_HUB palette) ──────────────────────────────────────
COLORS = {
    'primary':   '#38BDF8',
    'bg_dark':   '#0F172A',
    'card_dark': '#1E293B',
    'bg_light':  '#F1F5F9',
    'card_light':'#F8FAFC',
    'success':   '#10B981',
    'danger':    '#EF4444',
    'warn':      '#F59E0B',
    'muted':     '#64748B',
    'text_dark': '#F1F5F9',
    'text_light':'#0F172A',
}


def fmt_vnd(v, unit='T'):
    """Format số VNĐ sang đơn vị tỷ/triệu."""
    if unit == 'T':
        return f"{v/1e9:,.1f} Tỷ"
    return f"{v/1e6:,.1f} Tr"


def run():
    print("=" * 65)
    print("MODULE 8G — MASTER COMMAND CENTER DASHBOARD")
    print("=" * 65)
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── 1. Load CLEANED.csv ────────────────────────────────────────
    print("\n[1/6] Nạp CLEANED.csv cho KPI YTD...")
    df = pd.read_csv(CLEANED_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df.get('KẾT QUẢ'), errors='coerce').fillna(0)
    df['NỢ GỐC']  = pd.to_numeric(df.get('NỢ GỐC'),  errors='coerce').fillna(0)
    df['DPD']     = pd.to_numeric(df.get('DPD'),      errors='coerce')

    # Collapse to LOAN ID level
    loan = df.groupby('LOAN ID').agg(
        KQ_TONG   = ('KẾT QUẢ', 'sum'),
        NO_GOC    = ('NỢ GỐC',  'last'),
        DPD_CUOI  = ('DPD',     'last'),
        DU_AN     = ('DỰ ÁN',   'first'),
        CHI_NHANH = ('CHI NHÁNH','last'),
        AGENT     = ('PHỤ TRÁCH HỒ SƠ','last'),
    ).reset_index()
    loan['CÓ_THU'] = (loan['KQ_TONG'] > 0).astype(int)
    loan['REC_RATE'] = (loan['KQ_TONG'] / loan['NO_GOC'].replace(0, np.nan)).fillna(0)

    # KPI cốt lõi — 100% từ data thực
    ead_total  = loan['NO_GOC'].sum()
    ytd_total  = loan['KQ_TONG'].sum()
    arr_ytd    = ytd_total / ead_total if ead_total > 0 else 0
    n_total    = len(loan)
    n_paid     = loan['CÓ_THU'].sum()
    rpc_rate   = n_paid / n_total if n_total > 0 else 0

    print(f"   → Tổng EAD:     {fmt_vnd(ead_total)}")
    print(f"   → Kết quả YTD:  {fmt_vnd(ytd_total)}")
    print(f"   → ARR YTD:      {arr_ytd:.4%}")
    print(f"   → RPC Rate:     {rpc_rate:.4%}")

    # ── 2. Load DS_SEGMENTATION ────────────────────────────────────
    print("\n[2/6] Nạp DS_SEGMENTATION_FINAL.csv...")
    seg = None
    try:
        seg_chk  = pd.read_csv(DS_SEG_PATH, nrows=1, low_memory=False)
        seg_want = ['LOAN ID', 'CỤM_HÀNH_VI', 'PTP_SCORE_PERCENT',
                    'SỐ_NGÀY_KHÔNG_THANH_TOÁN', 'PHÂN LOẠI VÙNG MIỀN']
        seg_use  = [c for c in seg_want if c in seg_chk.columns]
        seg = pd.read_csv(DS_SEG_PATH, low_memory=False, usecols=seg_use)
        seg = seg.drop_duplicates(subset='LOAN ID', keep='last')
        loan = loan.merge(seg, on='LOAN ID', how='left')
        print(f"   → Đã gắn {len(seg_use)-1} cột AI")
    except Exception as e:
        print(f"   ⚠ Không tải DS_SEG: {e}")

    # ── 3. Load Feedback Loop ──────────────────────────────────────
    print("\n[3/6] Đọc Feedback Loop...")
    feedback_dpd   = 60
    feedback_active= False
    arr_snapshot   = {}
    triggered_cls  = []
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, encoding='utf-8') as fp:
                fb = json.load(fp)
            feedback_dpd    = fb.get('early_litigation_dpd', 60)
            arr_snapshot    = fb.get('arr_snapshot', {})
            triggered_cls   = fb.get('triggered_clusters', [])
            feedback_active = bool(triggered_cls)
            print(f"   ⚡ Feedback DPD: {feedback_dpd} ngày | Active: {feedback_active}")
        except Exception as e:
            print(f"   ⚠ {e}")
    else:
        print("   ℹ feedback_thresholds.json chưa có — chạy 8f trước")

    # ── 4. Phân tích Drill-down ────────────────────────────────────
    print("\n[4/6] Tính drill-down KPIs...")

    def agg_metrics(grp_col):
        if grp_col not in loan.columns:
            return pd.DataFrame()
        g = loan.groupby(grp_col, observed=True).agg(
            SỐ_HS   = ('LOAN ID', 'count'),
            KQ_SUM  = ('KQ_TONG', 'sum'),
            NG_SUM  = ('NO_GOC',  'sum'),
            N_PAID  = ('CÓ_THU',  'sum'),
        ).reset_index()
        g['ARR_%']  = (g['KQ_SUM'] / g['NG_SUM'].replace(0, np.nan)).fillna(0) * 100
        g['RPC_%']  = (g['N_PAID'] / g['SỐ_HS'].replace(0, np.nan)).fillna(0) * 100
        return g.sort_values('KQ_SUM', ascending=False)

    by_duan    = agg_metrics('DU_AN')
    by_chi     = agg_metrics('CHI_NHANH')
    by_cluster = agg_metrics('CỤM_HÀNH_VI') if 'CỤM_HÀNH_VI' in loan.columns else pd.DataFrame()

    # DPD band ARR
    dpd_bands = [(0,89),(90,180),(181,360),(361,540),(541,720),(721,1080),(1081,1440),(1441,1800),(1801,9999)]
    dpd_labels= ['0-89','90-180','181-360','361-540','541-720','721-1080','1081-1440','1441-1800','>1800']
    dpd_rows = []
    for (lo,hi), lbl in zip(dpd_bands, dpd_labels):
        sub = loan[loan['DPD_CUOI'].between(lo, hi)]
        n   = len(sub)
        kq  = sub['KQ_TONG'].sum()
        ng  = sub['NO_GOC'].sum()
        dpd_rows.append({'DPD_BAND': lbl, 'SỐ_HS': n,
                         'ARR_%': kq/ng*100 if ng>0 else 0,
                         'KQ_SUM': kq, 'NG_SUM': ng})
    dpd_df = pd.DataFrame(dpd_rows)

    # ── 5. Load Risk Foundations (nếu có từ 7d/7e) ───────────────
    print("\n[5/6] Load Risk Foundation data (7d LGD, 7e Roll-Rate)...")
    lgd_data   = None
    rr_data    = None
    try:
        lgd_data = pd.read_csv(LGD_CSV_PATH)
        print(f"   ✅ LGD data: {len(lgd_data):,} rows")
    except:
        print("   ℹ 7d_risk_lgd_model chưa chạy — Tab Risk sẽ hiển thị placeholder")
    try:
        rr_data = pd.read_csv(ROLLRATE_CSV, index_col=0)
        print(f"   ✅ Roll-Rate: {rr_data.shape}")
    except:
        print("   ℹ 7e_risk_roll_rate_matrix chưa chạy")

    # ── 6. Build Charts ────────────────────────────────────────────
    print("\n[6/6] Xây dựng Dashboard HTML...")

    # ── Tab 1 Charts ──────────────────────────────────────────────
    palette = px.colors.qualitative.Plotly

    # Chart: ARR by DỰ ÁN
    fig_duan = go.Figure()
    if not by_duan.empty:
        top_duan = by_duan.head(12)
        fig_duan.add_trace(go.Bar(
            x=top_duan['DU_AN'], y=top_duan['ARR_%'],
            marker_color=[COLORS['success'] if v >= arr_ytd*100 else COLORS['danger']
                         for v in top_duan['ARR_%']],
            text=[f"{v:.3f}%" for v in top_duan['ARR_%']],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>ARR: %{y:.4f}%<extra></extra>",
        ))
    fig_duan.update_layout(**_chart_layout("ARR (%) Theo Đối Tác (DỰ ÁN)", height=380))

    # Chart: ARR by DPD Band
    dpd_colors = [COLORS['success'] if v >= 1 else (COLORS['warn'] if v >= 0.1 else COLORS['danger'])
                  for v in dpd_df['ARR_%']]
    fig_dpd = go.Figure(go.Bar(
        x=dpd_df['DPD_BAND'], y=dpd_df['ARR_%'],
        marker_color=dpd_colors,
        text=[f"{v:.3f}%" for v in dpd_df['ARR_%']],
        textposition='outside',
        hovertemplate="<b>DPD %{x}</b><br>ARR: %{y:.4f}%<br>Số HS: %{customdata:,}<extra></extra>",
        customdata=dpd_df['SỐ_HS'],
    ))
    fig_dpd.update_layout(**_chart_layout("ARR (%) Theo Dải DPD — Aging Curve", height=380))

    # Chart: ARR by Cluster
    fig_cluster = go.Figure()
    if not by_cluster.empty:
        cls_colors = [palette[i % len(palette)] for i in range(len(by_cluster))]
        fig_cluster.add_trace(go.Bar(
            x=by_cluster['CỤM_HÀNH_VI'].astype(str), y=by_cluster['ARR_%'],
            marker_color=cls_colors,
            text=[f"{v:.4f}%" for v in by_cluster['ARR_%']],
            textposition='outside',
        ))
    fig_cluster.update_layout(**_chart_layout("ARR (%) Theo CỤM_HÀNH_VI (ML Cluster)", height=360))

    # Chart: Volume & RPC by Chi nhánh
    fig_branch = go.Figure()
    if not by_chi.empty:
        top_chi = by_chi[~by_chi['CHI_NHANH'].isin(['KHO','CLOSE CASE','nan'])].head(15)
        fig_branch.add_trace(go.Bar(
            name='RPC Rate (%)', x=top_chi['CHI_NHANH'], y=top_chi['RPC_%'],
            marker_color=COLORS['primary'],
            text=[f"{v:.2f}%" for v in top_chi['RPC_%']],
            textposition='outside',
        ))
    fig_branch.update_layout(**_chart_layout("RPC Rate (%) Theo Chi Nhánh", height=380))

    # ── Tab 2 (Risk) Charts ───────────────────────────────────────
    # LGD Gauge
    if lgd_data is not None and 'LGD' in lgd_data.columns:
        avg_lgd = lgd_data['LGD'].mean()
        total_el = lgd_data['EL_VND'].sum() if 'EL_VND' in lgd_data.columns else 0
    else:
        avg_lgd  = 1 - arr_ytd   # proxy: LGD ≈ 1 - ARR
        total_el = ead_total * avg_lgd

    fig_lgd = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=avg_lgd * 100,
        number={'suffix': '%', 'font': {'size': 42, 'color': COLORS['text_light']}},
        delta={'reference': 95, 'relative': False,
               'decreasing': {'color': COLORS['success']},
               'increasing': {'color': COLORS['danger']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['muted']},
            'bar':  {'color': COLORS['danger'] if avg_lgd > 0.8 else COLORS['warn']},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, 50],  'color': 'rgba(16,185,129,0.15)'},
                {'range': [50, 80], 'color': 'rgba(245,158,11,0.15)'},
                {'range': [80, 100],'color': 'rgba(239,68,68,0.15)'},
            ],
            'threshold': {'line': {'color': COLORS['primary'], 'width': 3}, 'value': 95},
        },
        title={'text': 'LGD Trung Bình (%)<br><span style="font-size:13px">Loss Given Default</span>',
               'font': {'size': 16}},
    ))
    fig_lgd.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=60, b=20, l=20, r=20),
                          font=dict(family='Inter, Arial, sans-serif'))

    # Roll-Rate Heatmap
    if rr_data is not None:
        rr_fig = go.Figure(go.Heatmap(
            z=rr_data.values,
            x=rr_data.columns.tolist(),
            y=rr_data.index.tolist() if hasattr(rr_data, 'index') else rr_data.iloc[:, 0].tolist(),
            colorscale='RdYlGn_r',
            texttemplate='%{z:.1f}%',
            colorbar=dict(title='Roll %'),
        ))
        rr_fig.update_layout(**_chart_layout("Roll-Rate Matrix — DPD Migration (%)", height=420))
    else:
        rr_fig = go.Figure()
        rr_fig.add_annotation(
            text="<b>⏳ Chưa có dữ liệu</b><br>Chạy 7e_risk_roll_rate_matrix.py để nạp dữ liệu",
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['muted'])
        )
        rr_fig.update_layout(**_chart_layout("Roll-Rate Matrix — DPD Migration (%)", height=420))

    # ARR by Cluster for Feedback Loop visualization
    feedback_rows = []
    for cls, arr in arr_snapshot.items():
        feedback_rows.append({'Cluster': cls, 'ARR': arr*100, 'Triggered': cls in triggered_cls})
    fb_df = pd.DataFrame(feedback_rows) if feedback_rows else pd.DataFrame(
        columns=['Cluster', 'ARR', 'Triggered'])

    fig_feedback = go.Figure()
    if not fb_df.empty:
        fig_feedback.add_trace(go.Bar(
            x=fb_df['Cluster'], y=fb_df['ARR'],
            marker_color=[COLORS['danger'] if t else COLORS['success'] for t in fb_df['Triggered']],
            text=[f"{v:.4f}%" for v in fb_df['ARR']],
            textposition='outside',
        ))
        fig_feedback.add_hline(y=0.5, line_dash='dash', line_color=COLORS['warn'],
                               annotation_text="Ngưỡng 0.5%", annotation_position="top right")
    fig_feedback.update_layout(**_chart_layout(
        "ARR Theo CỤM_HÀNH_VI (DPD 400-600) — Feedback Loop Monitor", height=360))

    # ── Build HTML Divs ───────────────────────────────────────────
    div = lambda fig: plo.plot(fig, output_type='div', include_plotlyjs=False)
    d_duan     = div(fig_duan)
    d_dpd      = div(fig_dpd)
    d_cluster  = div(fig_cluster)
    d_branch   = div(fig_branch)
    d_lgd      = div(fig_lgd)
    d_feedback = div(fig_feedback)
    d_rr       = div(rr_fig)  # always use rr_fig (handles both cases)

    # Feedback Loop status badge
    if feedback_active:
        fb_badge = f'''<div class="alert alert-danger">
            🚨 <strong>FEEDBACK LOOP KÍCH HOẠT</strong> — ARR &lt; 0.5% tại cụm: {", ".join(triggered_cls)}<br>
            ⇒ Ngưỡng khởi kiện sớm điều chỉnh: DPD {feedback_dpd} ngày
        </div>'''
    elif os.path.exists(FEEDBACK_PATH):
        fb_badge = f'<div class="alert alert-success">✅ ARR ổn định &gt; 0.5% — Giữ ngưỡng DPD mặc định ({feedback_dpd} ngày)</div>'
    else:
        fb_badge = '<div class="alert alert-info">ℹ Chưa có dữ liệu Feedback Loop. Chạy 8f để cập nhật.</div>'

    # ── Assemble Final HTML ───────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8G — Master Command Center Dashboard | VNE 2026</title>
<meta name="description" content="Bảng điều khiển trung tâm: ARR, RPC Rate, LGD, Roll-Rate, Feedback Loop — VNE Risk Management 2026">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg:      {COLORS['bg_light']};
    --card:    {COLORS['card_light']};
    --primary: #0F172A;
    --accent:  {COLORS['primary']};
    --success: {COLORS['success']};
    --danger:  {COLORS['danger']};
    --warn:    {COLORS['warn']};
    --muted:   {COLORS['muted']};
    --border:  #E2E8F0;
    --text:    {COLORS['text_light']};
    --sidebar: 210px;
    --nav-h:   48px;
    --radius:  14px;
    --shadow:  0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.05);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,.08), 0 4px 6px -4px rgba(0,0,0,.05);
}}
[data-theme="dark"] {{
    --bg:      {COLORS['bg_dark']};
    --card:    {COLORS['card_dark']};
    --primary: {COLORS['text_dark']};
    --border:  #334155;
    --text:    {COLORS['text_dark']};
    --muted:   #94A3B8;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior: smooth; }}
body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    transition: background .3s, color .3s;
}}

/* Custom scrollbars matching STRATEGY_HUB */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}

.sidebar::-webkit-scrollbar {{ width: 6px; }}

/* ── Top Nav ── */
.topnav {{
    position: fixed; top:0; left:0; right:0; height: var(--nav-h);
    background: var(--card);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; padding: 0 24px;
    gap: 16px; z-index: 100;
    box-shadow: var(--shadow);
}}
.topnav-brand {{
    font-size: 17px; font-weight: 700; color: var(--accent);
    letter-spacing: -.3px;
}}
.topnav-sub {{
    font-size: 12px; color: var(--muted); margin-left: 4px;
}}
.topnav-right {{ margin-left: auto; display:flex; gap:10px; align-items:center; }}
.badge-time {{
    font-size: 11px; color: var(--muted); background: var(--bg);
    padding: 4px 10px; border-radius: 20px; border: 1px solid var(--border);
}}
.btn-mode {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 12px; cursor: pointer;
    font-size: 12px; font-weight: 500; color: var(--text);
    transition: all .2s;
}}
.btn-mode:hover {{ border-color: var(--accent); color: var(--accent); }}
.back-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    color: var(--accent); text-decoration: none;
    font-size: 12px; font-weight: 600;
    padding: 5px 14px; border-radius: 8px;
    border: 1px solid var(--border);
    transition: 0.2s; white-space: nowrap;
}}
.back-btn:hover {{ background: var(--bg); }}

/* ── Sidebar ── */
.sidebar {{
    position: fixed; left:0; top: var(--nav-h); bottom:0;
    width: var(--sidebar);
    background: var(--card);
    border-right: 1px solid var(--border);
    padding: 25px 15px;
    overflow-y: auto; z-index: 90;
    box-sizing: border-box;
}}
.sidebar-section {{ margin-bottom: 8px; }}
.sidebar-label {{
    font-size: 9px; font-weight: 800; color: var(--muted);
    letter-spacing: 0.1em; margin: 15px 0 8px 0;
    text-transform: uppercase; padding: 0 12px;
}}
.sidebar-link {{
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 6px;
    font-size: 11px; font-weight: 600; color: var(--muted);
    text-decoration: none; cursor: pointer;
    transition: 0.2s;
    margin-bottom: 4px;
}}
.sidebar-link:hover {{ background: rgba(37,99,235,0.1); color: var(--accent); }}
.sidebar-link.active {{ background: rgba(37,99,235,0.1); color: var(--accent); }}
.sidebar-icon {{ font-size: 13px; width: 20px; text-align:center; }}

/* ── Main Content ── */
.main {{
    margin-left: var(--sidebar);
    margin-top: var(--nav-h);
    padding: 28px;
    flex: 1;
}}

/* ── Page Header ── */
.page-header {{
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #0F172A 100%);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow-lg);
}}
.page-header::after {{
    content:''; position:absolute; top:-40px; right:-40px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(56,189,248,.15) 0%, transparent 70%);
    border-radius: 50%;
}}
.page-header h1 {{
    color: #F1F5F9; font-size: 22px; font-weight: 800;
    letter-spacing: -.4px; margin-bottom: 6px;
}}
.page-header p {{ color: #94A3B8; font-size: 13px; }}
.header-chips {{ display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }}
.chip {{
    background: rgba(255,255,255,.08); color: #CBD5E1;
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 20px; padding: 4px 12px; font-size: 11px; font-weight: 500;
}}

/* ── KPI Strip ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 24px;
}}
.kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
    border-bottom: 4px solid var(--border);
    transition: transform .2s, box-shadow .2s;
    position: relative; overflow: hidden;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-lg); }}
.kpi-card.kpi-arr  {{ border-bottom-color: var(--accent); }}
.kpi-card.kpi-ead  {{ border-bottom-color: #8B5CF6; }}
.kpi-card.kpi-ytd  {{ border-bottom-color: var(--success); }}
.kpi-card.kpi-rpc  {{ border-bottom-color: var(--warn); }}
.kpi-card.kpi-lgd  {{ border-bottom-color: var(--danger); }}
.kpi-card.kpi-fb   {{ border-bottom-color: #EC4899; }}
.kpi-label {{
    font-size: 10px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .6px; margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 28px; font-weight: 800; color: var(--text);
    letter-spacing: -.5px; line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px; color: var(--muted); margin-top: 6px;
}}
.kpi-icon {{
    position: absolute; right:16px; top:16px;
    font-size: 28px; opacity: .12;
}}

/* ── Tab System ── */
.tab-bar {{
    display: flex; gap: 4px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 20px;
    width: fit-content;
}}
.tab-btn {{
    padding: 9px 22px; border-radius: 8px;
    border: none; background: transparent;
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 600;
    color: var(--muted); cursor: pointer;
    transition: all .2s;
}}
.tab-btn.active {{
    background: var(--card);
    color: var(--accent);
    box-shadow: var(--shadow);
}}
.tab-btn:hover:not(.active) {{ color: var(--text); }}
.tab-pane {{ display:none; }}
.tab-pane.active {{ display:block; animation: fadeUp .25s ease; }}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(6px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

/* ── Chart Cards ── */
.chart-grid-2 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px,1fr));
    gap: 20px; margin-bottom: 20px;
}}
.chart-grid-3 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px,1fr));
    gap: 20px; margin-bottom: 20px;
}}
.chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
}}
.chart-card.full {{ grid-column: 1/-1; }}
.chart-title {{
    font-size: 14px; font-weight: 700; color: var(--text);
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
}}

/* ── Alerts ── */
.alert {{
    padding: 14px 18px; border-radius: 10px; margin-bottom: 16px;
    border-left: 4px solid; font-size: 13px; line-height: 1.6;
}}
.alert-info    {{ background: #EFF6FF; border-color: #3B82F6; color: #1D4ED8; }}
.alert-success {{ background: #ECFDF5; border-color: var(--success); color: #065F46; }}
.alert-danger  {{ background: #FEF2F2; border-color: var(--danger); color: #991B1B; }}
.alert-warn    {{ background: #FFFBEB; border-color: var(--warn); color: #92400E; }}
[data-theme="dark"] .alert-info    {{ background: rgba(59,130,246,.12); color: #93C5FD; }}
[data-theme="dark"] .alert-success {{ background: rgba(16,185,129,.12); color: #6EE7B7; }}
[data-theme="dark"] .alert-danger  {{ background: rgba(239,68,68,.12);  color: #FCA5A5; }}
[data-theme="dark"] .alert-warn    {{ background: rgba(245,158,11,.12); color: #FCD34D; }}

/* ── Risk Section ── */
.risk-grid {{
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 20px; margin-bottom: 20px;
}}

/* ── Footer ── */
.footer {{
    margin-left: var(--sidebar);
    padding: 16px 28px;
    font-size: 11px; color: var(--muted);
    border-top: 1px solid var(--border);
    text-align: center;
}}
@media(max-width:768px) {{
    .sidebar {{ display:none; }}
    .main, .footer {{ margin-left:0; }}
    .risk-grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>


<!-- Top Navigation -->
<div class="topnav">
    <a href="../../STRATEGY_HUB.html" class="back-btn">← Strategy Hub</a>
    <span class="topnav-brand">⚖️ VNE — Quản trị Rủi ro</span>
    <span class="topnav-sub">| {now_str}</span>
    <div class="topnav-right">
        <button class="btn-mode" id="mode-btn" onclick="toggleTheme()">🌙 Mode</button>
    </div>
</div>

<!-- Sidebar -->
<aside class="sidebar">
    <div class="sidebar-section">
        <div class="sidebar-label">Bảng điều hành chính</div>
        <a class="sidebar-link active" onclick="switchTab('operational', this)" id="sidebar-btn-operational">
            <span class="sidebar-icon">📊</span> Giám sát Hoạt động
        </a>
        <a class="sidebar-link" onclick="switchTab('risk', this)" id="sidebar-btn-risk">
            <span class="sidebar-icon">🛡️</span> Giám sát Rủi ro
        </a>
    </div>
    <div class="sidebar-section" style="margin-top:20px;">
        <div class="sidebar-label">Phân tích Vận hành & Kết quả</div>
        <a class="sidebar-link" href="#" onclick="loadModule('8b_CONTACT_FUNNEL.html', this); return false;">
            <span class="sidebar-icon">📞</span> Phễu Tương tác & Kết nối khách hàng
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('8f_AGENT_PERFORMANCE.html', this); return false;">
            <span class="sidebar-icon">👤</span> Hiệu suất & Phân bổ Tác nghiệp
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('8a_AGING_CURVE_ANALYSIS.html', this); return false;">
            <span class="sidebar-icon">📈</span> Phân tích Tuổi nợ & Tỷ lệ Thu hồi
        </a>
    </div>
    <div class="sidebar-section" style="margin-top:20px;">
        <div class="sidebar-label">Cơ cấu Danh mục & Xu hướng</div>
        <a class="sidebar-link" href="#" onclick="loadModule('8d_GEO_RESIDUAL.html', this); return false;">
            <span class="sidebar-icon">🗺️</span> Địa lý & Sai lệch Vùng miền
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('8c_DEBT_STACKING.html', this); return false;">
            <span class="sidebar-icon">🔗</span> Tích tụ Nợ & Đa hồ sơ
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('8e_PRODUCT_VINTAGE.html', this); return false;">
            <span class="sidebar-icon">📦</span> Chất lượng Tín dụng theo Thế hệ (Vintage)
        </a>
    </div>
    <div class="sidebar-section" style="margin-top:20px;">
        <div class="sidebar-label">Mô hình Rủi ro & Dự báo chuyên sâu</div>
        <a class="sidebar-link" href="#" onclick="loadModule('7e_ROLL_RATE.html', this); return false;">
            <span class="sidebar-icon">🔄</span> Ma trận Dịch chuyển Nhóm nợ (Roll-Rate)
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('8i_RATING_MIGRATION.html', this); return false;">
            <span class="sidebar-icon">📊</span> Chuyển dịch Xếp hạng Tín dụng (Rating Migration)
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('7f_SURVIVAL_CURVES.html', this); return false;">
            <span class="sidebar-icon">⏳</span> Phân tích Thời gian Thanh toán (Survival Analysis)
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('7d_LGD_ANALYSIS.html', this); return false;">
            <span class="sidebar-icon">💰</span> Mô hình Tổn thất khi Vỡ nợ (LGD & Expected Loss)
        </a>
        <a class="sidebar-link" href="#" onclick="loadModule('8h_PARTNER_RISK.html', this); return false;">
            <span class="sidebar-icon">🤝</span> Phân rã & Tập trung Rủi ro Đối tác
        </a>
    </div>
    <div class="sidebar-section" style="margin-top:20px;">
        <div class="sidebar-label">Hệ thống</div>
        <div style="padding:10px 12px; font-size:11px; color:var(--muted); line-height:1.6;">
            <div>📂 155,302 hồ sơ</div>
            <div>🔄 Cập nhật: {now_str}</div>
            <div>⚙️ Framework v3.0</div>
        </div>
    </div>
</aside>

<!-- Main -->
<main class="main">

    <!-- Page Header -->
    <div class="page-header">
        <h1>Master Command Center Dashboard</h1>
        <p>Tổng quan danh mục thu hồi nợ — VNE 2026 | Powered by DS_SEGMENTATION + Feedback Loop</p>
        <div class="header-chips">
            <span class="chip">📊 {n_total:,} Hồ sơ</span>
            <span class="chip">💰 EAD: {fmt_vnd(ead_total)}</span>
            <span class="chip">✅ Kết quả: {fmt_vnd(ytd_total)}</span>
            <span class="chip">📐 ARR: {arr_ytd:.4%}</span>
            <span class="chip">{'🚨 Feedback Active' if feedback_active else '🟢 Feedback Normal'}</span>
        </div>
    </div>

    <!-- KPI Strip -->
    <div class="kpi-grid">
        <div class="kpi-card kpi-arr">
            <div class="kpi-icon">📐</div>
            <div class="kpi-label">ARR YTD (Tỷ lệ thu hồi nợ)</div>
            <div class="kpi-value">{arr_ytd:.4%}</div>
            <div class="kpi-sub">KẾT QUẢ / NỢ GỐC toàn danh mục</div>
        </div>
        <div class="kpi-card kpi-ead">
            <div class="kpi-icon">🏦</div>
            <div class="kpi-label">EAD — Tổng Nợ Gốc</div>
            <div class="kpi-value">{fmt_vnd(ead_total)}</div>
            <div class="kpi-sub">{n_total:,} hồ sơ đang quản lý</div>
        </div>
        <div class="kpi-card kpi-ytd">
            <div class="kpi-icon">💵</div>
            <div class="kpi-label">Kết Quả YTD</div>
            <div class="kpi-value">{fmt_vnd(ytd_total)}</div>
            <div class="kpi-sub">Tổng tiền thu được 3 tháng đầu năm</div>
        </div>
        <div class="kpi-card kpi-rpc">
            <div class="kpi-icon">📞</div>
            <div class="kpi-label">RPC Rate (Right Party Contact)</div>
            <div class="kpi-value">{rpc_rate:.2%}</div>
            <div class="kpi-sub">{n_paid:,} / {n_total:,} hồ sơ có kết quả</div>
        </div>
        <div class="kpi-card kpi-lgd">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-label">LGD Ước Tính</div>
            <div class="kpi-value">{avg_lgd:.1%}</div>
            <div class="kpi-sub">{'Từ 7d_LGD model' if lgd_data is not None else 'Proxy: 1 - ARR'}</div>
        </div>
        <div class="kpi-card kpi-fb">
            <div class="kpi-icon">🔄</div>
            <div class="kpi-label">Feedback Loop DPD Threshold</div>
            <div class="kpi-value">{feedback_dpd} ngày</div>
            <div class="kpi-sub">{'🚨 Kích hoạt — DPD giảm từ 60→30' if feedback_active else '✅ Ổn định — Ngưỡng mặc định'}</div>
        </div>
    </div>

    <!-- Tab Bar -->
    <div class="tab-bar">
        <button class="tab-btn active" id="btn-operational" onclick="switchTab('operational', this)">
            📊 Tab 1: Operational Monitor
        </button>
        <button class="tab-btn" id="btn-risk" onclick="switchTab('risk', this)">
            🛡️ Tab 2: Risk Monitor
        </button>
    </div>

    <!-- ═══ TAB 1: OPERATIONAL ═══ -->
    <div id="pane-operational" class="tab-pane active">
        {fb_badge}

        <div class="chart-grid-2">
            <div class="chart-card">
                <div class="chart-title">📦 ARR (%) Theo Đối Tác — Portfolio Contribution</div>
                {d_duan}
            </div>
            <div class="chart-card">
                <div class="chart-title">📈 ARR (%) Theo Dải DPD — Aging Curve View</div>
                {d_dpd}
            </div>
        </div>

        <div class="chart-grid-2">
            <div class="chart-card">
                <div class="chart-title">🤖 ARR (%) Theo ML Cluster (CỤM_HÀNH_VI)</div>
                {d_cluster}
            </div>
            <div class="chart-card">
                <div class="chart-title">👥 RPC Rate Theo Chi Nhánh</div>
                {d_branch}
            </div>
        </div>
    </div>

    <!-- ═══ TAB 2: RISK MONITOR ═══ -->
    <div id="pane-risk" class="tab-pane">
        <div class="alert alert-info">
            🛡️ <strong>Risk Monitor</strong> — LGD, Roll-Rate Matrix và Feedback Loop ARR Tracker.
            Dữ liệu LGD/Roll-Rate được nạp từ output của module <strong>7d</strong> và <strong>7e</strong>.
        </div>

        <div class="risk-grid">
            <div class="chart-card">
                <div class="chart-title">⚠️ LGD Gauge — Loss Given Default</div>
                {d_lgd}
                <div style="margin-top:12px; padding:12px; background:var(--bg); border-radius:8px; font-size:12px; color:var(--muted);">
                    <div>📌 Expected Loss = {fmt_vnd(total_el)}</div>
                    <div>📌 EL/EAD = {total_el/ead_total:.2%}</div>
                    <div>📌 Nguồn: {'7d_LGD Model' if lgd_data is not None else 'Proxy (1-ARR)'}</div>
                </div>
            </div>
            <div class="chart-card">
                <div class="chart-title">🔄 Roll-Rate Matrix — DPD Migration Probability</div>
                {d_rr}
            </div>
        </div>

        <div class="chart-card full">
            <div class="chart-title">📡 Feedback Loop ARR Monitor — CỤM_HÀNH_VI × DPD 400-600</div>
            {d_feedback}
            <div style="margin-top:10px; padding:10px 14px; background:var(--bg); border-radius:8px; font-size:12px; color:var(--muted);">
                <strong>Quy tắc Feedback Loop:</strong> Nếu ARR của CỤM_HÀNH_VI = 3 tại DPD 400-600 &lt; 0.5% trong 2 kỳ báo cáo liên tiếp 
                → tự động điều chỉnh ngưỡng khởi kiện sớm từ DPD 60 → DPD 30.
                Ngưỡng hiện tại: <strong>{feedback_dpd} ngày</strong>.
            </div>
        </div>
    </div>

</main>

<!-- Module Container for Dynamic Loading (SPA) -->
<div id="module-container" style="display:none; margin-left:var(--sidebar); margin-top:var(--nav-h); height:calc(100vh - var(--nav-h)); padding:0; box-sizing:border-box;">
    <iframe id="module-iframe" style="width:100%; height:100%; border:none;"></iframe>
</div>

<footer class="footer">
    Module 8G — Master Command Center Dashboard v1.0 | VNE Risk Management Framework 2026 | 
    Dữ liệu: TỔNG HỢP NĂM 2026 CLEANED.csv + DS_SEGMENTATION_FINAL.csv | {now_str}
</footer>

<script>
// ── Dark Mode Utilities ────────────────────────────────────────────
function updateCharts(isDark) {{
    const textCol = isDark ? '#F1F5F9' : '#334155';
    const gridCol = isDark ? 'rgba(51,65,85,.3)' : 'rgba(100,116,139,.15)';
    document.querySelectorAll('.plotly-graph-div').forEach(chart => {{
        try {{
            Plotly.relayout(chart, {{
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font.color': textCol,
                'xaxis.gridcolor': gridCol, 'yaxis.gridcolor': gridCol,
                'xaxis.tickfont.color': textCol, 'yaxis.tickfont.color': textCol,
                'legend.font.color': textCol,
            }});
        }} catch(e) {{}}
    }});
}}

function injectModuleTheme(iframe, isDark) {{
    try {{
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        if (!doc || !doc.head) return;
        const existing = doc.getElementById('vne-theme-override');
        if (existing) existing.remove();
        const style = doc.createElement('style');
        style.id = 'vne-theme-override';
        if (isDark) {{
            style.textContent = `:root {{--bg:#0F172A;--card:#1E293B;--border:#334155;--text:#F1F5F9;--primary:#38BDF8;--muted:#94A3B8;
                --background:#0F172A;--card-bg:#1E293B;--text-main:#F1F5F9;--text-muted:#94A3B8;}} 
                body{{background:#0F172A!important;color:#F1F5F9!important;}} 
                .section-card,.card,.chart-card,.kpi-card,.card-bg{{background:#1E293B!important;border-color:#334155!important;}} 
                .section-header,.header-section{{border-color:#334155!important;}} 
                th{{background:rgba(56,189,248,0.08)!important;color:#38BDF8!important;}} 
                td{{border-color:#334155!important;color:#F1F5F9!important;}} 
                tr:nth-child(even){{background:rgba(51,65,85,0.4)!important;}} 
                h1,h2,h3,h4,h5,h6{{color:#F1F5F9!important;}} 
                p,label,span:not(.badge){{color:inherit;}} 
                .alert-info{{background:rgba(59,130,246,.12)!important;color:#93C5FD!important;border-color:rgba(59,130,246,.3)!important;}} 
                .alert-success{{background:rgba(16,185,129,.12)!important;color:#6EE7B7!important;}} 
                .alert-danger,.alert-warning{{background:rgba(239,68,68,.12)!important;color:#FCA5A5!important;}} 
                input,select,textarea{{background:#1E293B!important;color:#F1F5F9!important;border-color:#334155!important;}}`;
            if (doc.body) doc.body.classList.add('dark-mode');
            doc.documentElement.setAttribute('data-theme', 'dark');
        }} else {{
            style.textContent = `:root {{--bg:#F8FAFC;--card:#FFFFFF;--border:#E2E8F0;--text:#1E293B;--primary:#2563EB;--muted:#64748B;
                --background:#F8FAFC;--card-bg:#FFFFFF;--text-main:#0F172A;--text-muted:#64748B;}} 
                body{{background:#F8FAFC!important;color:#1E293B!important;}} 
                .section-card,.card,.chart-card,.kpi-card,.card-bg{{background:#FFFFFF!important;border-color:#E2E8F0!important;}} 
                .section-header,.header-section{{border-color:#E2E8F0!important;}} 
                th{{background:rgba(37,99,235,0.05)!important;color:#1E3A8A!important;}} 
                td{{border-color:#E2E8F0!important;color:#1E293B!important;}} 
                tr:nth-child(even){{background:rgba(248,250,252,0.8)!important;}} 
                h1,h2,h3,h4,h5,h6{{color:#0F172A!important;}} 
                .alert-info{{background:#EFF6FF!important;color:#1D4ED8!important;}} 
                .alert-success{{background:#ECFDF5!important;color:#065F46!important;}} 
                .alert-danger,.alert-warning{{background:#FEF2F2!important;color:#991B1B!important;}} 
                input,select,textarea{{background:#FFFFFF!important;color:#1E293B!important;border-color:#E2E8F0!important;}}`;
            if (doc.body) doc.body.classList.remove('dark-mode');
            doc.documentElement.removeAttribute('data-theme');
        }}
        doc.head.appendChild(style);
    }} catch(e) {{ /* cross-origin or not loaded */ }}
}}

function toggleTheme() {{
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    const newDark = !isDark;
    if (newDark) {{ html.setAttribute('data-theme', 'dark'); }}
    else {{ html.removeAttribute('data-theme'); }}
    localStorage.setItem('vne_theme', newDark ? 'dark' : 'light');
    const btn = document.getElementById('mode-btn');
    if (btn) btn.textContent = newDark ? '☀️ Light' : '🌙 Mode';
    updateCharts(newDark);
    // Sync to any open sub-module
    const moduleIframe = document.getElementById('module-iframe');
    const container = document.getElementById('module-container');
    if (moduleIframe && container && container.style.display !== 'none') {{
        try {{ moduleIframe.contentWindow.postMessage({{ theme: newDark ? 'dark' : 'light' }}, '*'); }} catch(e) {{}}
        injectModuleTheme(moduleIframe, newDark);
        setTimeout(() => injectModuleTheme(moduleIframe, newDark), 300);
    }}
}}

// ── Module Loading (SPA sub-modules) ──────────────────────────────
function loadModule(url, btn) {{
    document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
    if (btn) btn.classList.add('active');
    document.querySelector('.main').style.display = 'none';
    const footer = document.querySelector('.footer');
    if (footer) footer.style.display = 'none';
    const container = document.getElementById('module-container');
    container.style.display = 'block';
    const iframe = document.getElementById('module-iframe');
    iframe.onload = () => {{
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        // Send postMessage for modules that have listeners
        try {{ iframe.contentWindow.postMessage({{ theme: isDark ? 'dark' : 'light' }}, '*'); }} catch(e) {{}}
        // Immediate CSS injection
        injectModuleTheme(iframe, isDark);
        // Delayed injection to win any race vs the module's own initTheme IIFE
        setTimeout(() => injectModuleTheme(iframe, isDark), 300);
    }};
    iframe.src = url;
}}

function switchTab(name, btn) {{
    const container = document.getElementById('module-container');
    if (container) container.style.display = 'none';
    document.querySelector('.main').style.display = 'block';
    const footer = document.querySelector('.footer');
    if (footer) footer.style.display = 'block';
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
    document.getElementById('pane-' + name).classList.add('active');
    document.getElementById('btn-' + name).classList.add('active');
    const sidebarBtn = document.getElementById('sidebar-btn-' + name);
    if (sidebarBtn) sidebarBtn.classList.add('active');
    if (btn) btn.classList.add('active');
    setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 100);
}}

// ── Theme Init (self-contained via localStorage) ───────────────────
function initTheme() {{
    const theme = localStorage.getItem('vne_theme');
    const isDark = theme === 'dark';
    if (isDark) {{ document.documentElement.setAttribute('data-theme', 'dark'); }}
    const btn = document.getElementById('mode-btn');
    if (btn) btn.textContent = isDark ? '☀️ Light' : '🌙 Mode';
    if (isDark) {{ setTimeout(() => updateCharts(true), 500); }}
}}
initTheme();

// Listen for theme-sync messages from child sub-module iframes
window.addEventListener('message', (e) => {{
    if (e.data && e.data.theme) {{
        // Only forward to nested module iframe (not propagate up)
        try {{
            const iframe = document.getElementById('module-iframe');
            if (iframe && iframe.src) {{
                iframe.contentWindow.postMessage({{ theme: e.data.theme }}, '*');
            }}
        }} catch(e) {{}}
    }}
}});
</script>
</body>
</html>"""

    out_path = os.path.join(REPORT_DIR, '8g_MASTER_DASHBOARD.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HOÀN THÀNH MODULE 8G!")
    print(f"   → HTML: {out_path}")
    print(f"   → ARR YTD:    {arr_ytd:.4%}")
    print(f"   → EAD:        {fmt_vnd(ead_total)}")
    print(f"   → KẾT QUẢ:   {fmt_vnd(ytd_total)}")
    print(f"   → Feedback:   DPD={feedback_dpd}d | Active={feedback_active}")


def _chart_layout(title, height=400):
    """Shared Plotly layout config (STRATEGY_HUB style, transparent bg)."""
    return dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=13, family='Inter, Arial', color='#64748B'), x=0),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Arial, sans-serif', size=11),
        margin=dict(t=40, b=40, l=50, r=40),
        xaxis=dict(gridcolor='rgba(100,116,139,.15)', zeroline=False, tickangle=-20),
        yaxis=dict(gridcolor='rgba(100,116,139,.15)', zeroline=False),
        showlegend=False,
    )


if __name__ == '__main__':
    run()
# -*- coding: utf-8 -*-
"""
MODULE 8F — HIỆU SUẤT NHÂN VIÊN CÓ HIỆU CHỈNH RỦI RO + FEEDBACK LOOP (AGENT PERFORMANCE + ARR TRIGGER)
Phiên bản: 2.0 (2026-05-26) — Tích hợp DS_SEG, tính ARR thực theo cụm, xuất Feedback Loop JSON
Câu hỏi: Agent/chi nhánh nào thực sự hiệu quả? ARR cụm 3 DPD 400-600 có ≤ 0.5% không?
Feedback Rule: Nếu ARR(Cụm 3, DPD 400-600) < 0.5% → xuất feedback_thresholds.json điều chỉnh DPD từ 60 → 30
Output:  reports/Data_Science/AGENT_PERFORMANCE.html
         reports/Data_Science/Data/feedback_thresholds.json
"""
import pandas as pd
import numpy as np
import os, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
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

FILE_PATH     = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'
DS_SEG_PATH   = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science\Data\DS_SEGMENTATION_FINAL.csv'
FEEDBACK_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science\Data\feedback_thresholds.json'

# Feedback Loop constants
ARR_THRESHOLD        = 0.005   # 0.5%
FEEDBACK_DPD_TRIGGER = 30      # Ngưỡng mới nếu khởi phát quá sớm
FEEDBACK_DPD_DEFAULT = 60      # Ngưỡng mặc định
FEEDBACK_DPD_RANGE   = (400, 600)  # Dải DPD để theo dõi

MIN_LOANS_PER_AGENT  = 30    # Agent cần ít nhất n HS để đủ ý nghĩa thống kê
MIN_LOANS_PER_BRANCH = 100

def run():
    print("=" * 60)
    print("MODULE 8F — AGENT PERFORMANCE ATTRIBUTION")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────
    print("\n[1/5] Nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df.get('KẾT QUẢ'), errors='coerce').fillna(0)
    df['DPD']     = pd.to_numeric(df.get('DPD'),      errors='coerce')
    df['NỢ GỐC']  = pd.to_numeric(df.get('NỢ GỐC'),  errors='coerce').fillna(0)

    # ── 2. Long → Wide per LOAN ID ────────────────────────────
    print("[2/5] Gom nhóm LOAN ID...")
    agg = df.groupby('LOAN ID').agg(
        KQ_TONG    = ('KẾT QUẢ',           'sum'),
        DPD_CUOI   = ('DPD',               'last'),
        NO_GOC     = ('NỢ GỐC',            'last'),
        AGENT      = ('PHỤ TRÁCH HỒ SƠ',  'last'),
        LEAD       = ('LEAD QUẢN LÝ HỒ SƠ','last'),
        CHI_NHANH  = ('CHI NHÁNH',         'last'),
        POS_NHOM   = ('PHÂN LOẠI POS',     'first'),
        TONG_DA_TT = ('TỔNG ĐÃ THANH TOÁN', 'last'),
    ).reset_index()

    # Merge DS_SEG cho CUM_HANH_VI + PTP_SCORE
    print("   Nạp DS_SEGMENTATION_FINAL.csv...")
    try:
        seg_chk  = pd.read_csv(DS_SEG_PATH, nrows=1, low_memory=False)
        seg_want = ['LOAN ID', 'CỤM_HÀNH_VI', 'PTP_SCORE_PERCENT', 'SỐ_NGÀY_KHÔNG_THANH_TOÁN']
        seg_use  = [c for c in seg_want if c in seg_chk.columns]
        seg_df   = pd.read_csv(DS_SEG_PATH, low_memory=False, usecols=seg_use)
        seg_df   = seg_df.drop_duplicates(subset='LOAN ID', keep='last')
        agg = agg.merge(seg_df, on='LOAN ID', how='left')
        print(f"   → Đã gắn {len(seg_use)-1} cột từ DS_SEG")
    except Exception as e:
        print(f"   ⚠ Không tải DS_SEG: {e}")

    agg['TARGET'] = (agg['KQ_TONG'] > 0).astype(int)
    agg = agg.dropna(subset=['DPD_CUOI'])
    print(f"   → {len(agg):,} LOAN ID")

    # ── 3. Baseline Model (loại trừ độ khó của hồ sơ) ─────────
    print("[3/5] Xây dựng mô hình baseline risk-adjustment...")
    pos_map = {'LOW POS': 0, 'MEDIUM POS': 1, 'HIGHT POS': 2}
    agg['POS_NUM'] = agg['POS_NHOM'].map(pos_map).fillna(1)
    feat_cols = ['DPD_CUOI', 'NO_GOC', 'POS_NUM']
    X_raw = agg[feat_cols].fillna(0)
    y     = agg['TARGET']

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_raw)
    lr      = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    lr.fit(X_sc, y)

    agg['BASELINE_PROB'] = lr.predict_proba(X_sc)[:, 1]
    agg['RESIDUAL']      = agg['TARGET'] - agg['BASELINE_PROB']

    # ── 4a. Phân tích theo Chi Nhánh ──────────────────────────
    print("[4/5] Phân tích residual theo Chi Nhánh & Agent...")
    branch_agg = agg.groupby('CHI_NHANH').agg(
        SO_HO_SO          = ('LOAN ID',      'count'),
        RESIDUAL_TB        = ('RESIDUAL',     'mean'),
        TY_LE_THUC_TE     = ('TARGET',       'mean'),
        EAD               = ('NO_GOC',       'sum'),
        TONG_THU          = ('KQ_TONG',      'sum')
    ).reset_index()
    branch_agg.rename(columns={'SO_HO_SO': 'SỐ_HỒ_SƠ', 'TY_LE_THUC_TE': 'TỶ_LỆ_THỰC_TẾ_%'}, inplace=True)
    branch_agg = branch_agg[branch_agg['SỐ_HỒ_SƠ'] >= MIN_LOANS_PER_BRANCH]
    branch_agg['RESIDUAL_TB']      = (branch_agg['RESIDUAL_TB']      * 100).round(3)
    branch_agg['TỶ_LỆ_THỰC_TẾ_%'] = (branch_agg['TỶ_LỆ_THỰC_TẾ_%'] * 100).round(2)
    branch_agg['EAD_TY']          = (branch_agg['EAD'] / 1e9).round(2)
    branch_agg['THU_TY']          = (branch_agg['TONG_THU'] / 1e9).round(3)
    branch_agg = branch_agg.sort_values('RESIDUAL_TB', ascending=False)

    # Loại bỏ giá trị lạ (KHO = chưa phân bổ)
    branch_agg = branch_agg[~branch_agg['CHI_NHANH'].isin(['KHO', 'CLOSE CASE'])]
    print("\n Chi Nhánh (Risk-adjusted Residual):")
    print(branch_agg.to_string(index=False))

    # ── 4b. Phân tích theo Agent ───────────────────────────────
    agent_agg = agg[~agg['AGENT'].isin(['KHO', 'nan', 'NaN'])].groupby('AGENT').agg(
        SỐ_HỒ_SƠ        = ('LOAN ID',  'count'),
        RESIDUAL_TB      = ('RESIDUAL', 'mean'),
        TỶ_LỆ_THỰC_TẾ   = ('TARGET',   'mean'),
        TỔNG_THU         = ('KQ_TONG',  'sum'),
        EAD             = ('NO_GOC',   'sum')
    ).reset_index()
    agent_agg = agent_agg[agent_agg['SỐ_HỒ_SƠ'] >= MIN_LOANS_PER_AGENT]
    agent_agg['RESIDUAL_TB']    = (agent_agg['RESIDUAL_TB']  * 100).round(3)
    agent_agg['TỶ_LỆ_THỰC_TẾ'] = (agent_agg['TỶ_LỆ_THỰC_TẾ'] * 100).round(2)
    agent_agg['EAD_TY']        = (agent_agg['EAD'] / 1e9).round(2)
    agent_agg['THU_TR']        = (agent_agg['TỔNG_THU'] / 1e6).round(1)
    agent_agg = agent_agg.sort_values('RESIDUAL_TB', ascending=False)

    print(f"\n Top 10 Super Agent (Residual cao nhất):")
    print(agent_agg.head(10)[['AGENT','SỐ_HỒ_SƠ','TỶ_LỆ_THỰC_TẾ','RESIDUAL_TB','TỔNG_THU']].to_string(index=False))

    # ANOVA test giữa các chi nhánh
    branch_groups = [
        agg.loc[agg['CHI_NHANH'] == b, 'RESIDUAL'].values
        for b in branch_agg['CHI_NHANH']
        if len(agg[agg['CHI_NHANH'] == b]) >= MIN_LOANS_PER_BRANCH
    ]
    try:
        f_stat, p_val = f_oneway(*branch_groups)
        sig_note = f"ANOVA F={f_stat:.2f}, p={p_val:.4f} → {'Sự khác biệt CÓ ý nghĩa thống kê' if p_val < 0.05 else 'Không có ý nghĩa thống kê'}"
        print(f"\n📊 {sig_note}")
    except Exception as e:
        sig_note = f"Không tính được ANOVA: {e}"
        print(f"⚠ {sig_note}")

    # ── 4b. Feedback Loop — Kiểm tra ARR thực tế ─────────────────────
    print("\n[4b/5] Kiểm tra Feedback Loop ARR...")
    import json
    dpd_lo, dpd_hi = FEEDBACK_DPD_RANGE
    feedback_triggered_clusters = []
    feedback_payload = {
        'early_litigation_dpd': FEEDBACK_DPD_DEFAULT,
        'triggered_clusters': [],
        'arr_snapshot': {},
        'threshold': ARR_THRESHOLD,
    }

    if 'CỤM_HÀNH_VI' in agg.columns:
        arr_by_cluster = agg[agg['DPD_CUOI'].between(dpd_lo, dpd_hi)].groupby('CỤM_HÀNH_VI', observed=True).agg(
            KQ  = ('KQ_TONG', 'sum'),
            NG  = ('NO_GOC',  'sum'),
            CNT = ('LOAN ID', 'count'),
        ).reset_index()
        arr_by_cluster['ARR'] = (arr_by_cluster['KQ'] / arr_by_cluster['NG'].replace(0, float('nan'))).fillna(0)

        print(f"\n   ARR thực tế theo CỤM_HÀNH_VI (DPD {dpd_lo}-{dpd_hi}):")
        for _, row in arr_by_cluster.iterrows():
            cum = row['CỤM_HÀNH_VI']
            arr = row['ARR']
            print(f"      {cum}: ARR={arr:.4%} ({int(row['CNT']):,} HS)")
            feedback_payload['arr_snapshot'][str(cum)] = round(float(arr), 6)
            if arr < ARR_THRESHOLD:
                feedback_triggered_clusters.append(str(cum))

        if feedback_triggered_clusters:
            feedback_payload['early_litigation_dpd'] = FEEDBACK_DPD_TRIGGER
            feedback_payload['triggered_clusters']   = feedback_triggered_clusters
            with open(FEEDBACK_PATH, 'w', encoding='utf-8') as fp:
                json.dump(feedback_payload, fp, ensure_ascii=False, indent=2)
            print(f"\n   🚨 FEEDBACK TRIGGERED: ARR < {ARR_THRESHOLD:.1%} tại cụm: {feedback_triggered_clusters}")
            print(f"      ⇒ Điều chỉnh early_litigation_dpd: {FEEDBACK_DPD_DEFAULT} → {FEEDBACK_DPD_TRIGGER}")
            print(f"      ⇒ Xuất: {FEEDBACK_PATH}")
        else:
            feedback_payload['early_litigation_dpd'] = FEEDBACK_DPD_DEFAULT
            with open(FEEDBACK_PATH, 'w', encoding='utf-8') as fp:
                json.dump(feedback_payload, fp, ensure_ascii=False, indent=2)
            print(f"   ✅ ARR ổn định trên {ARR_THRESHOLD:.1%} tại tất cả cụm — giữ ngưỡng mặc định {FEEDBACK_DPD_DEFAULT} ngày")
    else:
        print("   ⚠ Không có cột CỤM_HÀNH_VI — bỏ qua Feedback Loop")

    # ── 4c. Agent table (mở rộng) ────────────────────────────────
    agg['TỶ_LỆ_ĐÃ_THANH_TOÁN'] = (agg['TONG_DA_TT'].fillna(0) / agg['NO_GOC'].replace(0, float('nan'))).fillna(0)

    # ── 5. Visualization & Premium HTML Dashboard ────────────────
    print("\n[5/5] Tạo Dashboard HTML cao cấp...")
    palette = px.colors.qualitative.Plotly
    import plotly.offline as plo

    # Tab 1: Branch residual (fig1)
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Hiệu Suất Chi Nhánh (Residual %)", "Phân Bổ Hồ Sơ Theo Chi Nhánh"), specs=[[{"type": "bar"}, {"type": "pie"}]], horizontal_spacing=0.1)
    b_colors = ['#E53935' if v < 0 else '#1E88E5' for v in branch_agg['RESIDUAL_TB']]
    fig1.add_trace(go.Bar(x=branch_agg['CHI_NHANH'], y=branch_agg['RESIDUAL_TB'], marker_color=b_colors, text=[f"{v:+.3f}%" for v in branch_agg['RESIDUAL_TB']], textposition='outside'), row=1, col=1)
    fig1.add_hline(y=0, line_dash="dash", line_color="#FDD835", row=1, col=1)
    branch_vol = agg['CHI_NHANH'].value_counts().reset_index()
    branch_vol.columns = ['CHI_NHANH', 'SỐ_HS']
    fig1.add_trace(go.Pie(labels=branch_vol['CHI_NHANH'], values=branch_vol['SỐ_HS'], hole=0.4, textinfo='label+percent'), row=1, col=2)
    fig1.update_layout( height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40), showlegend=False)
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Tab 2: Agents (fig2)
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Top 15 Super Agent (Risk-adjusted Residual)", "Tỷ Lệ Thu Hồi Thực Tế vs Dự Báo (Chi Nhánh)"), horizontal_spacing=0.1)
    top_agents = agent_agg.head(15).sort_values('RESIDUAL_TB', ascending=True)
    fig2.add_trace(go.Bar(y=top_agents['AGENT'], x=top_agents['RESIDUAL_TB'], orientation='h', marker_color='#43A047', text=[f"{v:+.3f}%" for v in top_agents['RESIDUAL_TB']], textposition='outside'), row=1, col=1)
    fig2.add_trace(go.Bar(x=branch_agg['CHI_NHANH'], y=branch_agg['TỶ_LỆ_THỰC_TẾ_%'], marker_color='#1565C0', text=[f"{v:.2f}%" for v in branch_agg['TỶ_LỆ_THỰC_TẾ_%']], textposition='outside'), row=1, col=2)
    fig2.update_layout( height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40), showlegend=False)
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # Tab 3: Branch EAD & Portfolio (fig3)
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("EAD Tổng Theo Chi Nhánh (Tỷ VND)", "Tỷ Lệ Thu Hồi Trên EAD"), horizontal_spacing=0.1)
    fig3.add_trace(go.Bar(x=branch_agg['CHI_NHANH'], y=branch_agg['EAD_TY'], marker_color='#8B5CF6', text=[f"{v:.1f}T" for v in branch_agg['EAD_TY']], textposition='outside'), row=1, col=1)
    branch_agg['RECOVERY_EAD_%'] = (branch_agg['TONG_THU'] / branch_agg['EAD'] * 100).round(2)
    fig3.add_trace(go.Bar(x=branch_agg['CHI_NHANH'], y=branch_agg['RECOVERY_EAD_%'], marker_color='#F59E0B', text=[f"{v:.2f}%" for v in branch_agg['RECOVERY_EAD_%']], textposition='outside'), row=1, col=2)
    fig3.update_layout( height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40), showlegend=False)
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # Tab 4 Data Tables HTML rows
    branch_table_rows = ""
    for _, r in branch_agg.iterrows():
        res_color = "#10B981" if r['RESIDUAL_TB'] > 0 else "#EF4444"
        branch_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['CHI_NHANH']}</td>
            <td style="text-align:right;">{r['SỐ_HỒ_SƠ']:,}</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right;">{r['TỶ_LỆ_THỰC_TẾ_%']:.2f}%</td>
            <td style="text-align:right; font-weight:700; color:{res_color};">{r['RESIDUAL_TB']:+.3f}%</td>
            <td style="text-align:right;">{r['THU_TY']:.3f} Tỷ</td>
        </tr>"""

    agent_table_rows = ""
    for _, r in agent_agg.head(30).iterrows():
        res_color = "#10B981" if r['RESIDUAL_TB'] > 0 else "#EF4444"
        agent_table_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['AGENT']}</td>
            <td style="text-align:right;">{r['SỐ_HỒ_SƠ']:,}</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right;">{r['TỶ_LỆ_THỰC_TẾ']:.2f}%</td>
            <td style="text-align:right; font-weight:700; color:{res_color};">{r['RESIDUAL_TB']:+.3f}%</td>
            <td style="text-align:right;">{r['THU_TR']:.1f} Tr</td>
        </tr>"""

    # Insights HTML
    insights_html = ""
    top_branch = branch_agg.iloc[0]['CHI_NHANH'] if not branch_agg.empty else "N/A"
    top_branch_res = branch_agg.iloc[0]['RESIDUAL_TB'] if not branch_agg.empty else 0
    bot_branch = branch_agg.iloc[-1]['CHI_NHANH'] if not branch_agg.empty else "N/A"
    bot_branch_res = branch_agg.iloc[-1]['RESIDUAL_TB'] if not branch_agg.empty else 0
    
    insights_html += f"""
    <div class="alert-box alert-success">
        <strong>🟢 Chi nhánh dẫn đầu ({top_branch}):</strong> Hiệu suất thực tế vượt kỳ vọng {top_branch_res:+.3f}%. Chi nhánh này đang xử lý hồ sơ khó rất tốt.
    </div>"""
    
    if bot_branch_res < 0:
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>🔴 Chi nhánh tụt hậu ({bot_branch}):</strong> Hiệu suất thấp hơn kỳ vọng {bot_branch_res:+.3f}%. Cần kiểm tra lại chất lượng kịch bản call và nghiệp vụ nhân sự.
        </div>"""
        
    insights_html += f"""
    <div class="alert-box alert-warn">
        <strong>📊 Thống kê ý nghĩa (ANOVA):</strong> {sig_note}
    </div>"""

    if feedback_triggered_clusters:
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>⚠️ ARR Feedback Triggered:</strong> Cụm {feedback_triggered_clusters} có ARR thực tế < 0.5%. Đã xuất lệnh điều chỉnh DPD leo thang sớm (early_litigation_dpd) xuống {FEEDBACK_DPD_TRIGGER} ngày.
        </div>"""

    total_agents = len(agent_agg)
    avg_actual = branch_agg['TỶ_LỆ_THỰC_TẾ_%'].mean() if not branch_agg.empty else 0

    # HTML Layout
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8F — Agent Performance & ARR Feedback</title>
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
    <h1>8F — HIỆU SUẤT NHÂN VIÊN VÀ FEEDBACK LOOP (ARR)</h1>
    <p>Đánh giá hiệu suất đại lý/chi nhánh thông qua chỉ số Residual (sau khi chuẩn hoá độ khó hồ sơ) và tự động tạo Feedback điều chỉnh chiến lược theo ARR — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Số Chi Nhánh</div>
        <div class="kpi-value">{len(branch_agg)}</div>
        <div class="kpi-sub">Đạt chuẩn khối lượng phân tích</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Tổng Số Agent Cấp Cao</div>
        <div class="kpi-value">{total_agents}</div>
        <div class="kpi-sub">Nhân viên vượt mốc >30 hồ sơ</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Chi Nhánh Dẫn Đầu</div>
        <div class="kpi-value" style="font-size:18px;">{top_branch}</div>
        <div class="kpi-sub">Vượt kỳ vọng {top_branch_res:+.2f}%</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">ARR Trigger Status</div>
        <div class="kpi-value">{"ACTIVE" if feedback_triggered_clusters else "SAFE"}</div>
        <div class="kpi-sub">Giới hạn thời gian (Litigation DPD): {feedback_payload['early_litigation_dpd']} ngày</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Hiệu Suất Chi Nhánh</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. Bảng Xếp Hạng Agent</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. Nhận Định & Feedback Loop</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="chart-card">
        {div1}
    </div>
</div>

<div id="tab2" class="tab-content">
    <div class="chart-card">
        {div2}
    </div>
</div>

<div id="tab3" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Đánh Giá Hiệu Quả Thu Hồi (Risk-adjusted)</h3>
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

    out_html = os.path.join(REPORT_DIR, "8f_AGENT_PERFORMANCE.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv  = os.path.join(SUB_DATA_DIR, "8f_agent_performance.csv")
    agent_agg.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8F!")
    print(f"   → HTML:  {out_html}")
    print(f"   → CSV:   {out_csv}")
    print(f"   → Feedback: {FEEDBACK_PATH}")

if __name__ == "__main__":
    run()
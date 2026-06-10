# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

FILE_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\TỔNG HỢP NĂM 2026 CLEANED.csv'
OUTPUT_DIR = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Deep_Dive_Dashboards'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_engineer_features():
    df = pd.read_csv(FILE_PATH, low_memory=False)
    numeric_cols = ['KẾT QUẢ', 'NỢ GỐC', 'TỔNG NỢ', 'DPD', 'SỐ TIỀN GIẢI NGÂN', 'TỔNG ĐÃ THANH TOÁN', 'MỤC TIÊU VNE T01', 'MỤC TIÊU ĐỐI TÁC']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    if 'CLOSE CASE' not in df.columns:
        if 'TRẠNG THÁI' in df.columns:
            df['CLOSE CASE'] = df['TRẠNG THÁI'].apply(lambda x: 1 if str(x).upper() in ['CLOSE', 'ĐÓNG', 'HOÀN THÀNH'] else 0)
        else:
            df['CLOSE CASE'] = 0
            
    df = df.sort_values(['LOAN ID', 'THÁNG'])
    df['YTD_COLLECTED'] = df.groupby('LOAN ID')['KẾT QUẢ'].cumsum()
    # Xử lý Nợ gốc và Tổng nợ duy nhất tại thời điểm bàn giao
    initial_debt = df.drop_duplicates(subset=['LOAN ID'], keep='first')[['LOAN ID', 'NỢ GỐC', 'TỔNG NỢ']]
    initial_debt.columns = ['LOAN ID', 'ORIGINAL_PRINCIPAL', 'TOTAL_DEBT']
    df = df.merge(initial_debt, on='LOAN ID', how='left')
    def get_quarter(month_str):
        try:
            m = int(str(month_str).replace('THÁNG ', '').replace('T', '').split('.')[0])
            return f"QUÝ {(m-1)//3 + 1}"
        except: return "N/A"
    df['QUARTER'] = df['THÁNG'].apply(get_quarter)
    # Phân tách Hà Nội và TP.HCM
    def get_detailed_region(row):
        prov = str(row.get('TỈNH TẠM TRÚ', '')).upper()
        if 'HỒ CHÍ MINH' in prov: return 'HỒ CHÍ MINH'
        if 'HÀ NỘI' in prov: return 'HÀ NỘI'
        return row.get('PHÂN LOẠI VÙNG MIỀN', 'OTHER')
    
    df['PHÂN LOẠI VÙNG MIỀN'] = df.apply(get_detailed_region, axis=1)
    df_latest = df.drop_duplicates(subset=['LOAN ID'], keep='last').copy()
    return df, df_latest

def format_vn_hover(val):
    if val == 0: return '0'
    if abs(val) >= 1e12:
        v = val / 1e12
        return f"{v:.2f} NT"
    if abs(val) >= 1e9:
        v = val / 1e9
        return f"{v:.2f} T"
    if abs(val) >= 1e6:
        v = val / 1e6
        return f"{v:.0f} Tr"
    return f"{val:,.0f}"

def format_vn_axis(val):
    if val == 0: return '0'
    if abs(val) >= 1e12:
        v = val / 1e12
        return f"{v:.0f}NT" if v.is_integer() else f"{v:.1f}NT"
    if abs(val) >= 1e9:
        v = val / 1e9
        return f"{v:.0f}T" if v.is_integer() else f"{v:.1f}T"
    if abs(val) >= 1e6:
        v = val / 1e6
        return f"{v:.0f}Tr" if v.is_integer() else f"{v:.1f}Tr"
    return f"{val:,.0f}"

def get_linear_ticks(max_val):
    if max_val <= 0:
        return [0], ['0']
    steps = [
        1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6,
        1e9, 2e9, 5e9, 10e9, 20e9, 50e9, 100e9, 200e9, 500e9,
        1e12, 2e12, 5e12, 10e12, 20e12, 50e12, 100e12
    ]
    target_step = max_val / 5
    step = steps[0]
    for s in steps:
        if s >= target_step:
            step = s
            break
    tickvals = []
    val = 0
    while val <= max_val + step * 0.1:
        tickvals.append(val)
        val += step
    ticktext = [format_vn_axis(v) for v in tickvals]
    return tickvals, ticktext

def render_sidebar():
    return """
    <div class="sidebar">
        <nav class="sidebar-nav">
            <p class="nav-group-label">ĐIỀU HÀNH CHUNG</p>
            <a href="#kpis" class="sidebar-link">📊 Chỉ số Tổng quan</a>
            <a href="#partners" class="sidebar-link">🏢 Dự án Đối tác</a>
            <a href="#revenue" class="sidebar-link">📅 Doanh thu & Mục tiêu</a>
            
            <p class="nav-group-label">PHÂN TÍCH CHUYÊN SÂU</p>
            <a href="#leadership" class="sidebar-link">👥 Hiệu suất Lead</a>
            <a href="#trends" class="sidebar-link">📈 Xu hướng Productivity</a>
            <a href="#risk" class="sidebar-link">🚩 Rủi ro Trọng điểm</a>
            <a href="#geo" class="sidebar-link">📍 Phân khu Địa lý</a>
        </nav>
        <div class="sidebar-footer">
            <p>© 2026 VNE LAW FIRM</p>
        </div>
    </div>
    """

def render_nav(active_partner=None):
    partners = ["ABBANK", "BDI - SHB", "HANMIR - LOTTE", "HANMIR - MIRA", "LOTTE", "MC", "MSB", "SHB", "SVFC"]
    if not active_partner:
        links = "".join([f'<a href="Deep_Dive_Dashboards/DASHBOARD_{p.replace(" ", "_")}.html" class="nav-link">{p}</a>' for p in partners])
        nav_links = (
            '<a href="#" id="nav-btn-home" class="nav-link active" onclick="showHome(); return false;">TRANG CHỦ</a>'
            '<a href="#" id="nav-btn-risk" class="nav-link" onclick="showRisk(); return false;">QUẢN TRỊ RỦI RO</a>'
        )
    else:
        links = "".join([f'<a href="DASHBOARD_{p.replace(" ", "_")}.html" class="nav-link {"active" if p==active_partner else ""}">{p}</a>' for p in partners])
        nav_links = (
            '<a href="../STRATEGY_HUB.html" class="nav-link">TRANG CHỦ</a>'
            '<a href="../STRATEGY_HUB.html?view=risk" class="nav-link">QUẢN TRỊ RỦI RO</a>'
        )
    return f"""<div class="nav-container"><div class="nav-bar"><span class="brand">VNE LAW FIRM</span>{nav_links}<div style="flex:1"></div><button id="theme-toggle" style="background:none;border:1px solid var(--border);color:var(--primary);padding:4px 8px;border-radius:6px;cursor:pointer;font-size:14px;display:flex;align-items:center;gap:5px;font-weight:600">🌙 Mode</button></div><div class="nav-bar partners">{links}</div></div>"""

def render_kpi_cards(df_hist, df_latest):
    cases = df_latest['LOAN ID'].nunique()
    total_goc = df_latest['ORIGINAL_PRINCIPAL'].sum()
    total_thu = df_hist['KẾT QUẢ'].sum()
    ror = total_thu / total_goc if total_goc > 0 else 0
    total_target = df_hist['MỤC TIÊU VNE T01'].sum()
    pct_ta = total_thu / total_target if total_target > 0 else 0
    
    return f"""<div id="kpis" class="kpi-grid" style="grid-template-columns: repeat(5, 1fr);">
        <div class="kpi-card cases" style="border-bottom-color: #6366f1;"><span class="label">TỔNG SỐ HỒ SƠ</span><span class="value">{cases:,}</span></div>
        <div class="kpi-card principal"><span class="label">NỢ GỐC BÀN GIAO</span><span class="value">{total_goc:,.0f}<br>Đ</span></div>
        <div class="kpi-card collected"><span class="label">KẾT QUẢ (YTD)</span><span class="value">{total_thu:,.0f}<br>Đ</span></div>
        <div class="kpi-card rate"><span class="label">ROR</span><span class="value">{ror:.2%}</span></div>
        <div class="kpi-card target" style="border-bottom-color: #ec4899;"><span class="label">% TA</span><span class="value">{pct_ta:.1%}</span></div>
    </div>"""

def render_partner_section(df_latest):
    def get_dpd_bucket(dpd, project):
        if project == 'SVFC':
            if dpd < 360: return '<360'
            if dpd <= 540: return '360-540'
            if dpd <= 900: return '541-900'
            if dpd <= 1800: return '901-1800'
            return '>1800'
        elif project == 'MSB':
            if dpd < 30: return '<30'
            if dpd < 60: return '<60'
            if dpd < 90: return '<90'
            if dpd < 180: return '<180'
            if dpd < 360: return '<360'
            if dpd <= 720: return '361-720'
            if dpd <= 1800: return '720-1800'
            return '>1800'
        elif project == 'ABBANK':
            if dpd < 30: return '<30'
            if dpd < 60: return '<60'
            if dpd < 90: return '<90'
            if dpd < 180: return '<180'
            if dpd < 360: return '<360'
            if dpd <= 720: return '361-720'
            if dpd <= 1080: return '721-1080'
            return '>1080'
        else: # NHÓM CHUNG
            if dpd < 360: return '<360'
            if dpd < 1000: return '<1000'
            if dpd < 1800: return '<1800'
            return '>1800'

    p_data = df_latest.groupby('DỰ ÁN').agg({"LOAN ID": "count", "ORIGINAL_PRINCIPAL": "sum", "YTD_COLLECTED": "sum", "DPD": "mean"}).reset_index()
    p_data.columns = ['DỰ ÁN', 'CASE_COUNT', 'ORIGINAL_PRINCIPAL', 'YTD_COLLECTED', 'AVG_DPD']
    p_data["AVG_POS_VAL"] = (p_data["ORIGINAL_PRINCIPAL"] / p_data["CASE_COUNT"].replace(0, 1)).fillna(0)
    total_cases = p_data['CASE_COUNT'].sum()
    p_data["RECOVERY_RATE"] = (p_data["YTD_COLLECTED"] / p_data["ORIGINAL_PRINCIPAL"].replace(0, np.nan)).fillna(0)
    p_data["CASE_SHARE"] = (p_data["CASE_COUNT"] / total_cases).fillna(0)
    p_data = p_data.sort_values("CASE_COUNT", ascending=False)
    cards_html = ""
    for _, r in p_data.iterrows():
        dpd_val = r['AVG_DPD']; dpd_color = "#10b981"
        if dpd_val > 720: dpd_color = "#ef4444"
        elif dpd_val > 360: dpd_color = "#f59e0b"
        elif dpd_val > 180: dpd_color = "#fcd34d"
        
        # Chi tiết DPD & Đánh giá
        proj_df = df_latest[df_latest['DỰ ÁN'] == r['DỰ ÁN']]
        buckets = proj_df['DPD'].apply(lambda x: get_dpd_bucket(x, r['DỰ ÁN'])).value_counts().to_dict()
        b_html = " | ".join([f"{k}: {v}" for k, v in buckets.items()])
        
        ratings = proj_df['ĐÁNH GIÁ KHÁCH HÀNG'].value_counts().to_dict() if 'ĐÁNH GIÁ KHÁCH HÀNG' in proj_df.columns else {}
        r_html = " | ".join([f"{k}: {v}" for k, v in sorted(ratings.items()) if str(k) in ['0', 'A', 'B', 'C', 'D']])
        
        cards_html += f"""<a href="Deep_Dive_Dashboards/DASHBOARD_{r['DỰ ÁN'].replace(' ', '_')}.html" class="partner-card">
            <div class="p-header">
                <span class="p-name">{r['DỰ ÁN']}</span>
                <span style="font-size:11px;color:#94a3b8;font-weight:600;display:inline-block;white-space:nowrap">POS: {r['ORIGINAL_PRINCIPAL']:,.0f} Đ</span>
                <span class="p-dpd" style="color:#ef4444;display:inline-block;white-space:nowrap">Avg POS: {r['AVG_POS_VAL']:,.0f} Đ</span>
            </div>
            <div class="p-row"><span class="p-cases">{r['CASE_COUNT']:,} HS ({r['CASE_SHARE']:.1%})</span><span class="p-rate"><small style="font-size:12px;color:#64748b;font-weight:400;margin-right:4px">ROR</small>{r['RECOVERY_RATE']:.2%}</span></div>
            <div class="p-bar-bg" style="height:22px;background:#f1f5f9;border-radius:11px;margin:12px 0;position:relative;overflow:hidden;border:1px solid #e2e8f0;box-shadow:inset 0 1px 2px rgba(0,0,0,0.05)">
                <div class="p-bar-fill" style="width:{min(r['RECOVERY_RATE']*100, 100)}%;height:100%;transition:0.3s;background:linear-gradient(90deg, #3b82f6, #6366f1);box-shadow:0 0 10px rgba(59,130,246,0.2)"></div>
                <div style="position:absolute;width:100%;top:0;left:0;height:100%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;color:#1e293b;pointer-events:none;letter-spacing:0.02em;white-space:nowrap">
                    {r['YTD_COLLECTED']:,.0f} Đ / {r['ORIGINAL_PRINCIPAL']:,.0f} Đ
                </div>
            </div>
            <div style="margin-top:12px;padding-top:8px;border-top:1px solid #e2e8f0;font-size:9px">
                <div style="color:#64748b;margin-bottom:4px"><b>DPD:</b> {b_html}</div>
                <div style="color:#2563eb"><b>Cơ cấu đánh giá KH:</b> {r_html}</div>
            </div></a>"""
    return f'<div id="partners" class="lua-chon-label">📍 Lựa chọn Dự án Điều hành</div><div class="partner-grid-full">{cards_html}</div>'

def render_revenue_section(df_hist):
    # Logic nợ gốc và tổng nợ duy nhất
    def get_unique_metric(idx, col):
        return df_hist.loc[idx].drop_duplicates('LOAN ID')[col].sum()

    m_data = df_hist.groupby('THÁNG').agg({
        'LOAN ID': 'nunique',
        'ORIGINAL_PRINCIPAL': lambda x: get_unique_metric(x.index, 'ORIGINAL_PRINCIPAL'),
        'TOTAL_DEBT': lambda x: get_unique_metric(x.index, 'TOTAL_DEBT'),
        'KẾT QUẢ': 'sum',
        'MỤC TIÊU VNE T01': 'sum',
        'MỤC TIÊU ĐỐI TÁC': 'sum'
    }).reset_index().sort_values('THÁNG')
    
    m_data['RATE'] = (m_data['KẾT QUẢ'] / (m_data['ORIGINAL_PRINCIPAL'].replace(0, np.nan))).fillna(0)
    
    fig = go.Figure()
    
    def format_vn_hover(val):
        if abs(val) >= 1e12: return f"{val/1e12:.2f} NT"
        if abs(val) >= 1e9: return f"{val/1e9:.2f} T"
        if abs(val) >= 1e6: return f"{val/1e6:.0f} Tr"
        return f"{val:,.0f}"
    
    # 1. Cột Kết quả thực thu (Cam) - Dịch xuống dưới
    fig.add_trace(go.Bar(
        name='Kết quả thực thu',
        x=m_data['THÁNG'],
        y=m_data['KẾT QUẢ'],
        marker_color='#f97316',
        text=m_data['KẾT QUẢ'].apply(lambda x: f"{x/1e9:.1f}T"),
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=11, weight='bold', color='#ffffff'),
        customdata=m_data['KẾT QUẢ'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))

    # 2. Cột Nợ gốc bàn giao (Xanh) - Xếp lên trên
    fig.add_trace(go.Bar(
        name='Nợ gốc bàn giao',
        x=m_data['THÁNG'],
        y=m_data['ORIGINAL_PRINCIPAL'],
        marker_color='#3b82f6',
        text=m_data['ORIGINAL_PRINCIPAL'].apply(lambda x: f"{x/1e9:.1f}T"),
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=11, weight='bold', color='#ffffff'),
        customdata=m_data['ORIGINAL_PRINCIPAL'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))

    # 3. Đường Mục tiêu VNE (Đường đứt nét màu đỏ)
    fig.add_trace(go.Scatter(
        name='Mục tiêu VNE',
        x=m_data['THÁNG'],
        y=m_data['MỤC TIÊU VNE T01'],
        mode='lines+markers',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(size=7, color='white', line=dict(color='#ef4444', width=2)),
        customdata=m_data['MỤC TIÊU VNE T01'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))

    # 4. Đường Mục tiêu Đối tác (Đường nét liền đỏ đậm)
    fig.add_trace(go.Scatter(
        name='Mục tiêu Đối tác',
        x=m_data['THÁNG'],
        y=m_data['MỤC TIÊU ĐỐI TÁC'],
        mode='lines+markers',
        line=dict(color='#b91c1c', width=3, dash='solid'),
        marker=dict(size=7, color='white', line=dict(color='#b91c1c', width=2)),
        customdata=m_data['MỤC TIÊU ĐỐI TÁC'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white", # Light mode template
        height=400,
        margin=dict(l=20,r=20,t=40,b=60),
        barmode='stack',
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color="#0f172a"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        xaxis=dict(type='category', title=None),
        title=dict(text="THỐNG KÊ DOANH THU THEO THÁNG", x=0.5, font=dict(size=14, weight='bold', color='#0f172a'))
    )
    
    fig.update_yaxes(
        title_text="Giá trị tiền (VNĐ) - LOG SCALE",
        showgrid=False,
        type='log',
        tickmode='array',
        tickvals=[1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14],
        ticktext=['1Tr', '10Tr', '100Tr', '1T', '10T', '100T', '1NT', '10NT', '100NT']
    )
    
    fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
    
    def build_table_rows(data):
        rows = ""
        for _, r in data.iterrows():
            ror = r['KẾT QUẢ'] / r['ORIGINAL_PRINCIPAL'] if r['ORIGINAL_PRINCIPAL'] > 0 else 0
            
            p_target = r['MỤC TIÊU ĐỐI TÁC']
            if p_target > 0:
                p_pct = r['KẾT QUẢ'] / p_target
                p_pct_str = f"{p_pct:.1%}"
                p_color = 'text-success' if p_pct >= 1 else 'text-danger'
            else:
                p_pct_str = " - "
                p_color = "opacity-50"
            
            v_target = r['MỤC TIÊU VNE T01']
            if v_target > 0:
                v_pct = r['KẾT QUẢ'] / v_target
                v_pct_str = f"{v_pct:.1%}"
                v_color = 'text-success' if v_pct >= 1 else 'text-danger'
            else:
                v_pct_str = " - "
                v_color = "opacity-50"
            
            rows += f"<tr><td>{r.iloc[0]}</td><td class='text-right font-bold'>{r['LOAN ID']:,}</td><td class='text-right'>{r['ORIGINAL_PRINCIPAL']:,.0f}</td><td class='text-right'>{r['TOTAL_DEBT']:,.0f}</td><td class='text-right' style='color: #fbbf24; font-weight: 800;'>{r['MỤC TIÊU ĐỐI TÁC']:,.0f}</td><td class='text-right text-warn font-bold'>{r['MỤC TIÊU VNE T01']:,.0f}</td><td class='text-right font-bold text-primary'>{r['KẾT QUẢ']:,.0f}</td><td class='text-right text-success font-bold'>{ror:.2%}</td><td class='text-right {p_color} font-bold'>{p_pct_str}</td><td class='text-right {v_color} font-bold'>{v_pct_str}</td></tr>"
        
        # Hàng TỔNG (Phải xử lý unique cho số lượng hồ sơ và nợ gốc)
        total_loan_id = df_hist['LOAN ID'].nunique()
        total_original_principal = df_hist.drop_duplicates('LOAN ID')['ORIGINAL_PRINCIPAL'].sum()
        total_debt = df_hist.drop_duplicates('LOAN ID')['TOTAL_DEBT'].sum()
        total_p_target = data['MỤC TIÊU ĐỐI TÁC'].sum()
        total_v_target = data['MỤC TIÊU VNE T01'].sum()
        total_result = data['KẾT QUẢ'].sum()
        
        t_ror = total_result / total_original_principal if total_original_principal > 0 else 0
        
        if total_p_target > 0:
            t_p_pct = total_result / total_p_target
            t_p_pct_str = f"{t_p_pct:.1%}"
            t_p_color = 'text-success' if t_p_pct >= 1 else 'text-danger'
        else:
            t_p_pct_str = " - "
            t_p_color = ""
            
        if total_v_target > 0:
            t_v_pct = total_result / total_v_target
            t_v_pct_str = f"{t_v_pct:.1%}"
            t_v_color = 'text-success' if t_v_pct >= 1 else 'text-danger'
        else:
            t_v_pct_str = " - "
            t_v_color = ""
        
        footer = f"""<tr style="border-top: 2px solid var(--primary); background: rgba(56,189,248,0.05); font-weight: 800;">
            <td>TỔNG</td>
            <td class="text-right">{total_loan_id:,}</td>
            <td class="text-right">{total_original_principal:,.0f}</td>
            <td class="text-right">{total_debt:,.0f}</td>
            <td class="text-right" style="color: #fbbf24;">{total_p_target:,.0f}</td>
            <td class="text-right" style="color:var(--warn)">{total_v_target:,.0f}</td>
            <td class="text-right" style="color:var(--primary)">{total_result:,.0f}</td>
            <td class="text-right text-success">{t_ror:.2%}</td>
            <td class="text-right {t_p_color}">{t_p_pct_str}</td>
            <td class="text-right {t_v_color}">{t_v_pct_str}</td>
        </tr>"""
        return rows, footer

    m_rows, m_footer = build_table_rows(m_data)
    
    p_data = df_hist.groupby('DỰ ÁN').agg({
        'LOAN ID': 'nunique',
        'ORIGINAL_PRINCIPAL': lambda x: get_unique_metric(x.index, 'ORIGINAL_PRINCIPAL'),
        'TOTAL_DEBT': lambda x: get_unique_metric(x.index, 'TOTAL_DEBT'),
        'KẾT QUẢ': 'sum',
        'MỤC TIÊU VNE T01': 'sum',
        'MỤC TIÊU ĐỐI TÁC': 'sum'
    }).reset_index().sort_values('KẾT QUẢ', ascending=False)
    
    # Tạo biểu đồ Dự án
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        name='Kết quả thực thu',
        x=p_data['DỰ ÁN'],
        y=p_data['KẾT QUẢ'],
        marker_color='#f97316',
        text=p_data['KẾT QUẢ'].apply(lambda x: f"{x/1e9:.1f}T"),
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=10, weight='bold', color='#ffffff'),
        customdata=p_data['KẾT QUẢ'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))
    fig_p.add_trace(go.Bar(
        name='Nợ gốc bàn giao',
        x=p_data['DỰ ÁN'],
        y=p_data['ORIGINAL_PRINCIPAL'],
        marker_color='#3b82f6',
        text=p_data['ORIGINAL_PRINCIPAL'].apply(lambda x: f"{x/1e9:.1f}T"),
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=10, weight='bold', color='#ffffff'),
        customdata=p_data['ORIGINAL_PRINCIPAL'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))
    fig_p.add_trace(go.Scatter(
        name='Mục tiêu VNE',
        x=p_data['DỰ ÁN'],
        y=p_data['MỤC TIÊU VNE T01'],
        mode='lines+markers',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(size=7, color='white', line=dict(color='#ef4444', width=2)),
        customdata=p_data['MỤC TIÊU VNE T01'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))
    fig_p.add_trace(go.Scatter(
        name='Mục tiêu Đối tác',
        x=p_data['DỰ ÁN'],
        y=p_data['MỤC TIÊU ĐỐI TÁC'],
        mode='lines+markers',
        line=dict(color='#b91c1c', width=3, dash='solid'),
        marker=dict(size=7, color='white', line=dict(color='#b91c1c', width=2)),
        customdata=p_data['MỤC TIÊU ĐỐI TÁC'].apply(format_vn_hover),
        hovertemplate="%{fullData.name}: %{customdata}<extra></extra>"
    ))
    
    fig_p.update_layout(template="plotly_white", height=400, margin=dict(l=20,r=20,t=40,b=60), barmode='stack', legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), hoverlabel=dict(bgcolor="white", font_size=13, font_color="#0f172a"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", xaxis=dict(type='category', title=None), title=dict(text="THỐNG KÊ DOANH THU THEO DỰ ÁN", x=0.5, font=dict(size=14, weight='bold', color='#0f172a')))
    fig_p.update_yaxes(
        title_text="Giá trị tiền (VNĐ) - LOG SCALE",
        showgrid=False,
        type='log',
        tickmode='array',
        tickvals=[1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14],
        ticktext=['1Tr', '10Tr', '100Tr', '1T', '10T', '100T', '1NT', '10NT', '100NT']
    )
    fig_p_html = fig_p.to_html(full_html=False, include_plotlyjs=False)

    p_rows, p_footer = build_table_rows(p_data)
    
    return f"""<div id="revenue" class="section-card"><div class="section-header"><h3>📅 PHÂN TÍCH DOANH THU & THEO DÕI MỤC TIÊU</h3></div><div style="margin-bottom: 30px; background: rgba(56,189,248,0.01); border-radius: 12px; padding: 15px;">{fig_html}</div><div style="display: flex; flex-direction: column; gap: 35px;"><div><h4 class="table-title">PHÂN TÍCH CHI TIẾT</h4><div class="table-container"><table><thead><tr><th>Tháng</th><th class='text-right'>Hồ sơ</th><th class='text-right'>Nợ gốc</th><th class='text-right'>Tổng nợ</th><th class='text-right'>Mục tiêu đối tác</th><th class='text-right'>VNE Target</th><th class='text-right'>Kết quả</th><th class='text-right'>ROR</th><th class='text-right'>% Partner Target</th><th class='text-right'>% VNE Target</th></tr></thead><tbody>{m_rows}</tbody><tfoot>{m_footer}</tfoot></table></div></div><div style="background: rgba(56,189,248,0.01); border-radius: 12px; padding: 15px; margin-top:20px;">{fig_p_html}</div><div><h4 class="table-title">PHÂN TÍCH DỰ ÁN</h4><div class="table-container"><table><thead><tr><th>Dự án</th><th class='text-right'>Hồ sơ</th><th class='text-right'>Nợ gốc</th><th class='text-right'>Tổng nợ</th><th class='text-right'>Mục tiêu đối tác</th><th class='text-right'>VNE Target</th><th class='text-right'>Kết quả</th><th class='text-right'>ROR</th><th class='text-right'>% Partner Target</th><th class='text-right'>% VNE Target</th></tr></thead><tbody>{p_rows}</tbody><tfoot>{p_footer}</tfoot></table></div></div></div></div><style>.table-title{{color:var(--primary); font-size:11px; font-weight:800; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em;}}</style>"""

def render_team_performance(df_hist, df_latest):
    keys = ['Quangnv', 'Sangnn', 'Tridt', 'Tuoint']
    total_cases_all = df_hist['LOAN ID'].nunique()
    
    monthly_team = df_hist[df_hist['LEAD QUẢN LÝ HỒ SƠ'].isin(keys)].groupby(['THÁNG', 'LEAD QUẢN LÝ HỒ SƠ'])['KẾT QUẢ'].sum().reset_index()
    
    max_val = monthly_team['KẾT QUẢ'].max() if not monthly_team.empty else 0
    tickvals, ticktext = get_linear_ticks(max_val)
    
    monthly_team['TEXT_LABEL'] = monthly_team['KẾT QUẢ'].apply(format_vn_axis)
    
    fig = px.bar(monthly_team, x='THÁNG', y='KẾT QUẢ', color='LEAD QUẢN LÝ HỒ SƠ', barmode='group', text='TEXT_LABEL')
    fig.update_layout(template="plotly_white", height=280, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext,
        title_text="Giá trị (VND)"
    )
    
    for trace in fig.data:
        lead_name = trace.name
        trace_data = monthly_team[monthly_team['LEAD QUẢN LÝ HỒ SƠ'] == lead_name]
        trace.customdata = trace_data['KẾT QUẢ'].apply(format_vn_hover)
        trace.hovertemplate = "<b>%{x}</b><br>" + f"Lead {lead_name}: " + "%{customdata}<extra></extra>"
        
    fig.update_traces(textposition='inside', insidetextanchor='middle', textfont=dict(size=9, weight='bold', color='#ffffff'))
    bar_html = fig.to_html(full_html=False, include_plotlyjs=False)
    
    def get_unique_metric_hist(idx, col):
        return df_hist.loc[idx].drop_duplicates('LOAN ID')[col].sum()

    # Tổng hợp dữ liệu cho tất cả các Lead
    team_full = df_hist.groupby('LEAD QUẢN LÝ HỒ SƠ').agg({
        'LOAN ID': 'nunique',
        'ORIGINAL_PRINCIPAL': lambda x: get_unique_metric_hist(x.index, 'ORIGINAL_PRINCIPAL'),
        'TOTAL_DEBT': lambda x: get_unique_metric_hist(x.index, 'TOTAL_DEBT'),
        'KẾT QUẢ': 'sum',
        'MỤC TIÊU VNE T01': 'sum',
        'MỤC TIÊU ĐỐI TÁC': 'sum'
    }).reset_index().sort_values('KẾT QUẢ', ascending=False)

    def build_team_rows(data, is_special=False):
        rows = ""
        for _, r in data.iterrows():
            ror = r['KẾT QUẢ'] / r['ORIGINAL_PRINCIPAL'] if r['ORIGINAL_PRINCIPAL'] > 0 else 0
            hs_pct = r['LOAN ID'] / total_cases_all if total_cases_all > 0 else 0
            
            p_target = r['MỤC TIÊU ĐỐI TÁC']
            p_pct_str = f"{r['KẾT QUẢ'] / p_target:.1%}" if p_target > 0 else " - "
            p_color = 'text-success' if (p_target > 0 and r['KẾT QUẢ'] / p_target >= 1) else 'text-danger' if p_target > 0 else "opacity-50"
            
            v_target = r['MỤC TIÊU VNE T01']
            v_pct_str = f"{r['KẾT QUẢ'] / v_target:.1%}" if v_target > 0 else " - "
            v_color = 'text-success' if (v_target > 0 and r['KẾT QUẢ'] / v_target >= 1) else 'text-danger' if v_target > 0 else "opacity-50"
            
            style = "background: rgba(16, 185, 129, 0.1); color: var(--success);" if is_special else ""
            rows += f"<tr style='{style}'><td>{r.iloc[0]}</td><td class='text-right'>{r['LOAN ID']:,}</td><td class='text-right font-bold text-primary'>{hs_pct:.1%}</td><td class='text-right'>{r['ORIGINAL_PRINCIPAL']:,.0f}</td><td class='text-right'>{r['TOTAL_DEBT']:,.0f}</td><td class='text-right' style='color: #fbbf24; font-weight: 800;'>{r['MỤC TIÊU ĐỐI TÁC']:,.0f}</td><td class='text-right text-warn font-bold'>{r['MỤC TIÊU VNE T01']:,.0f}</td><td class='text-right font-bold text-primary'>{r['KẾT QUẢ']:,.0f}</td><td class='text-right text-success font-bold'>{ror:.2%}</td><td class='text-right {p_color} font-bold'>{p_pct_str}</td><td class='text-right {v_color} font-bold'>{v_pct_str}</td></tr>"
        return rows

    main_rows = build_team_rows(team_full)

    # Tính toán hàng CLOSE CASE
    df_close = df_hist[df_hist['CLOSE CASE'] == 1]
    if not df_close.empty:
        close_df = pd.DataFrame([{
            'Lead': 'CLOSE CASE',
            'LOAN ID': df_close['LOAN ID'].nunique(),
            'ORIGINAL_PRINCIPAL': df_close.drop_duplicates('LOAN ID')['ORIGINAL_PRINCIPAL'].sum(),
            'TOTAL_DEBT': df_close.drop_duplicates('LOAN ID')['TOTAL_DEBT'].sum(),
            'KẾT QUẢ': df_close['KẾT QUẢ'].sum(),
            'MỤC TIÊU VNE T01': df_close['MỤC TIÊU VNE T01'].sum(),
            'MỤC TIÊU ĐỐI TÁC': df_close['MỤC TIÊU ĐỐI TÁC'].sum()
        }])
        close_row = build_team_rows(close_df, is_special=True)
    else:
        close_row = ""

    return f"""<div id="leadership" class="section-card">
        <div class="section-header"><h3>👥 HIỆU SUẤT ĐIỀU HÀNH (LEADERSHIP)</h3></div>
        <div style="margin-bottom: 25px;"><p style="color: #94a3b8; font-size: 11px; margin-bottom: 10px;">📊 RANKING THỰC THU TỪNG TEAM THEO THÁNG (GIÁ TRỊ CHUẨN)</p>{bar_html}</div>
        <div>
            <p class="table-title">PHÂN TÍCH HIỆU SUẤT ĐỘI NGŨ</p>
            <div class="table-container">
                <table>
                    <thead>
                        <tr><th>Lead</th><th class='text-right'>Hồ sơ</th><th class='text-right'>% HS</th><th class='text-right'>Nợ gốc</th><th class='text-right'>Tổng nợ</th><th class='text-right'>Mục tiêu đối tác</th><th class='text-right'>VNE Target</th><th class='text-right'>Kết quả</th><th class='text-right'>ROR</th><th class='text-right'>% Partner Target</th><th class='text-right'>% VNE Target</th></tr>
                    </thead>
                    <tbody>
                        {main_rows}
                        {close_row}
                    </tbody>
                </table>
            </div>
        </div>
    </div>"""

def render_trend_module(df_hist):
    col = 'MÃ TÌNH TRẠNG LIÊN HỆ'
    if col not in df_hist.columns:
        return f"""<div id="trends" class="section-card"><div class="section-header"><h3>📈 CHIẾN LƯỢC XU HƯỚNG (PRODUCTIVITY)</h3></div><p>Thiếu dữ liệu: {col}</p></div>"""
    
    unique_months = df_hist['THÁNG'].dropna().unique()
    if len(unique_months) == 0:
        return f"""<div id="trends" class="section-card"><div class="section-header"><h3>📈 CHIẾN LƯỢC XU HƯỚNG (PRODUCTIVITY)</h3></div><p>Thiếu dữ liệu tháng</p></div>"""
    
    def parse_month_num(m_str):
        try:
            clean = str(m_str).upper().replace('THÁNG ', '').replace('T', '').strip()
            parts = clean.split('.')
            m = int(parts[0])
            if len(parts) > 1:
                try:
                    y = int(parts[1])
                    return y * 12 + m
                except:
                    pass
            return m
        except:
            return 0
            
    sorted_months = sorted(unique_months, key=parse_month_num)
    latest_month = sorted_months[-1]
    prev_month = sorted_months[-2] if len(sorted_months) > 1 else None
    
    # Tính toán phễu cho tháng mới nhất
    df_latest_m = df_hist[df_hist['THÁNG'] == latest_month].copy()
    total_loans = df_latest_m['LOAN ID'].nunique()
    
    df_latest_m['CODE_CLEAN'] = df_latest_m[col].astype(str).str.upper().str.strip()
    df_latest_m['CODE_CLEAN'] = df_latest_m['CODE_CLEAN'].replace('NPS', 'NSP')
    
    valid_latest = df_latest_m[~df_latest_m['CODE_CLEAN'].isin(['NAN', '0', 'NONE', ''])]
    latest_code = valid_latest.groupby('LOAN ID')['CODE_CLEAN'].last()
    
    t0_n = len(latest_code)
    never_touched = total_loans - t0_n
    
    # Ring 2: Connected & beyond (T2+)
    T2_PLUS = {'CBACK', 'HUP', 'LM', 'NEGO', 'BPTP', 'NIOP', 'NCAP', 'FRAUD', 'KK/ĐXKK', 'PTP', 'PAID'}
    t2_plus_n = latest_code.isin(T2_PLUS).sum()
    
    # Ring 3: Processed & beyond (T3+)
    T3_PLUS = {'NEGO', 'BPTP', 'NIOP', 'NCAP', 'FRAUD', 'KK/ĐXKK', 'PTP', 'PAID'}
    t3_plus_n = latest_code.isin(T3_PLUS).sum()
    
    # Ring 4: Committed & beyond (T4+)
    T4_PLUS = {'PTP', 'PAID'}
    t4_plus_n = latest_code.isin(T4_PLUS).sum()
    
    # Ring 5: Resolved (T5)
    T5 = {'PAID'}
    t5_n = latest_code.isin(T5).sum()
    
    pct_t0 = (t0_n / total_loans) * 100 if total_loans > 0 else 0
    pct_t2 = (t2_plus_n / total_loans) * 100 if total_loans > 0 else 0
    pct_t3 = (t3_plus_n / total_loans) * 100 if total_loans > 0 else 0
    pct_t4 = (t4_plus_n / total_loans) * 100 if total_loans > 0 else 0
    pct_t5 = (t5_n / total_loans) * 100 if total_loans > 0 else 0
    
    # Spin Index tháng mới nhất
    total_actions_m = len(df_latest_m)
    spin_m = total_actions_m / total_loans if total_loans > 0 else 0
    
    # Render Multi-layer Progress Rings bằng Plotly
    fig = go.Figure()
    stages = [
        {'v': pct_t0, 'c': '#1e3a8a', 'label': f'T0: COVERAGE ({t0_n:,} HS)'},
        {'v': pct_t2, 'c': '#3b82f6', 'label': f'T2+: REACH ({t2_plus_n:,} HS)'},
        {'v': pct_t3, 'c': '#eab308', 'label': f'T3+: PROCESS ({t3_plus_n:,} HS)'},
        {'v': pct_t4, 'c': '#10b981', 'label': f'T4+: COMMIT ({t4_plus_n:,} HS)'},
        {'v': pct_t5, 'c': '#ef4444', 'label': f'T5: RESOLVE ({t5_n:,} HS)'}
    ]
    
    for i, stage in enumerate(stages):
        margin = i * 0.08
        fig.add_trace(go.Pie(
            values=[stage['v'], max(0, 100 - stage['v'])],
            labels=[stage['label'], "Chưa đạt"],
            hole=0.82,
            domain={'x': [0 + margin, 1 - margin], 'y': [0 + margin, 1 - margin]},
            marker=dict(colors=[stage['c'], 'rgba(148, 163, 184, 0.08)']),
            showlegend=False,
            hoverinfo='label+percent',
            textinfo='none',
            sort=False,
            direction='clockwise'
        ))
        
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[{
            'text': f"<span style='font-size:24px; font-weight:900;'>#Spin {spin_m:.2f}</span><br><span style='font-size:12px; color:#64748b;'>Tần suất tác động TB</span>",
            'showarrow': False,
            'x': 0.5, 'y': 0.5
        }]
    )
    
    radial_html = fig.to_html(full_html=False, include_plotlyjs=False)
    
    # Tính toán bảng biến động
    df_prev = df_hist[df_hist['THÁNG'] == prev_month].copy() if prev_month else pd.DataFrame()
    if not df_prev.empty:
        df_prev['CODE_CLEAN'] = df_prev[col].astype(str).str.upper().str.strip()
        df_prev['CODE_CLEAN'] = df_prev['CODE_CLEAN'].replace('NPS', 'NSP')
        valid_prev = df_prev[~df_prev['CODE_CLEAN'].isin(['NAN', '0', 'NONE', ''])]
        codes_prev = valid_prev.groupby('LOAN ID')['CODE_CLEAN'].last()
    else:
        codes_prev = pd.Series(dtype=str)
        
    all_loans = set(df_prev['LOAN ID'].unique()).union(set(df_latest_m['LOAN ID'].unique())) if prev_month else set(df_latest_m['LOAN ID'].unique())
    loan_weights = df_hist.drop_duplicates('LOAN ID').set_index('LOAN ID')['ORIGINAL_PRINCIPAL'].to_dict()
    
    transitions = []
    for loan_id in all_loans:
        c_prev = codes_prev.get(loan_id, 'CHƯA TÁC ĐỘNG')
        c_latest = latest_code.get(loan_id, 'CHƯA TÁC ĐỘNG')
        weight = loan_weights.get(loan_id, 0)
        transitions.append({
            'PREV_CODE': c_prev,
            'LATEST_CODE': c_latest,
            'WEIGHT': weight,
            'LOAN ID': loan_id
        })
        
    df_trans = pd.DataFrame(transitions)
    if not df_trans.empty:
        trans_agg = df_trans.groupby(['PREV_CODE', 'LATEST_CODE']).agg(
            DOSSIERS=('LOAN ID', 'count'),
            TOTAL_WEIGHT=('WEIGHT', 'sum')
        ).reset_index()
        trans_agg = trans_agg[~((trans_agg['PREV_CODE'] == 'CHƯA TÁC ĐỘNG') & (trans_agg['LATEST_CODE'] == 'CHƯA TÁC ĐỘNG'))]
        trans_agg = trans_agg.sort_values(by='DOSSIERS', ascending=False)
        # Giới hạn tối đa 50 cặp biến động
        trans_agg = trans_agg.head(50)
    else:
        trans_agg = pd.DataFrame()
        
    def get_code_badge(code):
        code = str(code).upper().strip()
        if code == 'CHƯA TÁC ĐỘNG':
            return '<span class="badge-trans badge-trans-gray">Chưa TĐ</span>'
        elif code == 'PAID':
            return '<span class="badge-trans badge-trans-success">PAID</span>'
        elif code == 'PTP':
            return '<span class="badge-trans badge-trans-info">PTP</span>'
        elif code in ['BPTP', 'HUP']:
            return f'<span class="badge-trans badge-trans-danger">{code}</span>'
        elif code == 'NEGO':
            return f'<span class="badge-trans badge-trans-warning">{code}</span>'
        elif code in ['NCON', 'NSP']:
            return f'<span class="badge-trans badge-trans-secondary">{code}</span>'
        elif code in ['NCAP', 'KK/ĐXKK', 'FRAUD', 'NIOP']:
            label = code
            if label == 'KK/ĐXKK': label = 'PLÝ'
            return f'<span class="badge-trans badge-trans-dark" title="{code}">{label}</span>'
        else:
            return f'<span class="badge-trans badge-trans-light">{code}</span>'
            
    table_rows = ""
    prev_m_label = prev_month.replace('THÁNG ', 'T') if prev_month else 'N/A'
    latest_m_label = latest_month.replace('THÁNG ', 'T')
    
    if trans_agg.empty:
        table_rows = f"<tr><td colspan='5' class='text-center text-muted'>Không có dữ liệu biến động</td></tr>"
    else:
        for _, row in trans_agg.iterrows():
            badge_prev = get_code_badge(row['PREV_CODE'])
            badge_latest = get_code_badge(row['LATEST_CODE'])
            weight_str = format_vn_axis(row['TOTAL_WEIGHT'])
            table_rows += f"""<tr>
                <td>{badge_prev}</td>
                <td style="text-align: center; color: var(--muted);">→</td>
                <td>{badge_latest}</td>
                <td class='text-right font-bold'>{row['DOSSIERS']:,} HS</td>
                <td class='text-right text-primary font-bold'>{weight_str}</td>
            </tr>"""
            
    legend_items = f"""
        <div style='display:flex; flex-direction:column; gap:10px;'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:12px; height:12px; background:#1e3a8a; border-radius:3px;'></div>
                <div style='display:flex; flex-direction:column;'>
                    <span style='color:var(--text); font-size:10.5px; font-weight:800;'>T0: COVERAGE — {pct_t0:.1f}% ({t0_n:,} HS)</span>
                    <span style='color:var(--muted); font-size:9px;'>Tỉ lệ bao phủ đã tác động / tổng danh mục ({total_loans:,} HS)</span>
                </div>
            </div>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:12px; height:12px; background:#3b82f6; border-radius:3px;'></div>
                <div style='display:flex; flex-direction:column;'>
                    <span style='color:var(--text); font-size:10.5px; font-weight:800;'>T2+: REACH — {pct_t2:.1f}% ({t2_plus_n:,} HS)</span>
                    <span style='color:var(--muted); font-size:9px;'>Tỉ lệ kết nối tiếp cận thành công</span>
                </div>
            </div>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:12px; height:12px; background:#eab308; border-radius:3px;'></div>
                <div style='display:flex; flex-direction:column;'>
                    <span style='color:var(--text); font-size:10.5px; font-weight:800;'>T3+: PROCESS — {pct_t3:.1f}% ({t3_plus_n:,} HS)</span>
                    <span style='color:var(--muted); font-size:9px;'>Tỉ lệ đang đàm phán & xử lý (SOFT/HARD)</span>
                </div>
            </div>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:12px; height:12px; background:#10b981; border-radius:3px;'></div>
                <div style='display:flex; flex-direction:column;'>
                    <span style='color:var(--text); font-size:10.5px; font-weight:800;'>T4+: COMMIT — {pct_t4:.1f}% ({t4_plus_n:,} HS)</span>
                    <span style='color:var(--muted); font-size:9px;'>Tỉ lệ cam kết hứa thanh toán (PTP)</span>
                </div>
            </div>
            <div style='display:flex; align-items:center; gap:12px;'>
                <div style='width:12px; height:12px; background:#ef4444; border-radius:3px;'></div>
                <div style='display:flex; flex-direction:column;'>
                    <span style='color:var(--text); font-size:10.5px; font-weight:800;'>T5: RESOLVE — {pct_t5:.1f}% ({t5_n:,} HS)</span>
                    <span style='color:var(--muted); font-size:9px;'>Tỉ lệ thanh toán thực tế (PAID)</span>
                </div>
            </div>
        </div>
    """

    return f"""<div id="trends" class="section-card">
        <style>
        .trend-grid {{
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 30px;
            align-items: start;
        }}
        @media (max-width: 1024px) {{
            .trend-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .trans-title {{
            color: var(--primary);
            font-size: 11px;
            font-weight: 800;
            margin-top: 20px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .trans-table-container {{
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--card);
            margin-top: 10px;
            margin-bottom: 20px;
        }}
        .trans-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10.5px;
        }}
        .trans-table th {{
            position: sticky;
            top: 0;
            background: var(--card);
            padding: 8px 10px;
            text-align: left;
            border-bottom: 2px solid var(--border);
            color: var(--muted);
            font-weight: 800;
            z-index: 10;
        }}
        .trans-table td {{
            padding: 6px 10px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        .trans-table tr:hover {{
            background: rgba(37, 99, 235, 0.02);
        }}
        .badge-trans {{
            display: inline-block;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 9px;
            font-weight: 800;
            text-transform: uppercase;
            text-align: center;
            min-width: 48px;
            box-sizing: border-box;
        }}
        .badge-trans-gray {{ background: #cbd5e1; color: #334155; }}
        .badge-trans-success {{ background: #d1fae5; color: #065f46; }}
        .badge-trans-info {{ background: #e0f2fe; color: #0369a1; }}
        .badge-trans-danger {{ background: #fee2e2; color: #991b1b; }}
        .badge-trans-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-trans-secondary {{ background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }}
        .badge-trans-dark {{ background: #1e293b; color: #f1f5f9; }}
        .badge-trans-light {{ background: #f8fafc; color: #64748b; }}

        body.dark-mode .badge-trans-gray {{ background: #334155; color: #cbd5e1; }}
        body.dark-mode .badge-trans-success {{ background: #065f46; color: #a7f3d0; }}
        body.dark-mode .badge-trans-info {{ background: #0369a1; color: #bae6fd; }}
        body.dark-mode .badge-trans-danger {{ background: #991b1b; color: #fecaca; }}
        body.dark-mode .badge-trans-warning {{ background: #92400e; color: #fde68a; }}
        body.dark-mode .badge-trans-secondary {{ background: #334155; color: #cbd5e1; border: 1px solid #475569; }}
        body.dark-mode .badge-trans-dark {{ background: #e2e8f0; color: #1e293b; }}
        body.dark-mode .badge-trans-light {{ background: #1e293b; color: #94a3b8; }}
        </style>
        
        <div class="section-header"><h3>📈 CHIẾN LƯỢC XU HƯỚNG (PRODUCTIVITY)</h3></div>
        <div class="trend-grid">
            <div style="position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(37, 99, 235, 0.01); border-radius: 16px; border: 1px solid var(--border); padding: 15px;">
                <p style="color: var(--primary); font-size: 11px; text-transform: uppercase; font-weight: 800; margin-bottom: 10px; letter-spacing: 0.05em;">Sơ đồ phễu hiệu suất đồng tâm ({latest_month})</p>
                <div style="width: 100%; display: flex; align-items: center; justify-content: center;">
                    {radial_html}
                </div>
            </div>
            
            <div style="background: var(--card); border: 1px solid var(--border); padding: 25px; border-radius: 16px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h4 style="color: var(--primary); font-size: 13px; font-weight: 800; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">💡 HIỆU SUẤT VẬN HÀNH</h4>
                    {legend_items}
                    
                    <p class="trans-title">🔄 Biến động trạng thái ({prev_m_label} → {latest_m_label})</p>
                    <div class="trans-table-container">
                        <table class="trans-table">
                            <thead>
                                <tr>
                                    <th>{prev_m_label}</th>
                                    <th style="width: 20px; text-align: center;"></th>
                                    <th>{latest_m_label}</th>
                                    <th class='text-right'>Hồ sơ</th>
                                    <th class='text-right'>Dư nợ gốc</th>
                                </tr>
                            </thead>
                            <tbody>
                                {table_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div style="margin-top: 10px; padding: 15px; background: rgba(37, 99, 235, 0.04); border-radius: 12px; border-left: 4px solid var(--success);">
                    <p style="color: var(--text); font-size: 11px; margin: 0; line-height: 1.6;">
                        🎯 <b>Chiến lược:</b> Sự thu hẹp khoảng cách giữa các vòng màu thể hiện <b>tỉ lệ rơi rụng (Drop-off)</b>. Bảng biến động trạng thái cho thấy xu hướng dịch chuyển của dòng tiền và hành vi khách hàng qua các tháng.
                    </p>
                </div>
            </div>
        </div>
    </div>"""

def render_risk_section(df_latest):
    risk_df = df_latest.sort_values('ORIGINAL_PRINCIPAL', ascending=False).copy()
    top_10 = risk_df.head(10)
    top_20 = risk_df.head(20)
    max_val = top_20['ORIGINAL_PRINCIPAL'].max() if not top_20.empty else 0
    tickvals, ticktext = get_linear_ticks(max_val)
    
    fig_bar = px.bar(top_20, x='LOAN ID', y='ORIGINAL_PRINCIPAL', color='ORIGINAL_PRINCIPAL', 
                     text=top_20['ORIGINAL_PRINCIPAL'].apply(format_vn_axis), color_continuous_scale='Reds')
    fig_bar.update_layout(template="plotly_white", height=300, margin=dict(l=10,r=10,t=10,b=10), 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, coloraxis_showscale=False)
    fig_bar.update_yaxes(
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext,
        title_text="Dư nợ gốc",
        range=[0, max_val * 1.15]
    )
    fig_bar.update_traces(
        textposition='outside', 
        textfont=dict(size=9, weight='bold'),
        customdata=top_20['ORIGINAL_PRINCIPAL'].apply(format_vn_hover),
        hovertemplate="Hồ sơ %{x}<br>Dư nợ gốc: %{customdata}<extra></extra>"
    )
    bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False)
    risk_df = risk_df.sort_values('ORIGINAL_PRINCIPAL', ascending=False)
    risk_df['cumulative_debt'] = risk_df['ORIGINAL_PRINCIPAL'].cumsum()
    risk_df['cumulative_percent'] = 100 * risk_df['cumulative_debt'] / risk_df['ORIGINAL_PRINCIPAL'].sum()
    risk_df['idx'] = range(1, len(risk_df) + 1)
    risk_df['idx_percent'] = 100 * risk_df['idx'] / len(risk_df)
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(x=risk_df['idx_percent'], y=risk_df['cumulative_percent'], name="Sự tập trung nợ", fill='tozeroy', line=dict(color='#ef4444', width=3)))
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="#94a3b8", annotation_text="Ngưỡng 80%")
    fig_pareto.update_layout(template="plotly_white", height=250, margin=dict(l=20,r=20,t=10,b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="% Số h/sơ", yaxis_title="% Tổng nợ")
    pareto_html = fig_pareto.to_html(full_html=False, include_plotlyjs=False)
    rows = "".join([f"<tr><td>{r['LOAN ID']}</td><td>{r['DỰ ÁN']}</td><td class='text-right'>{r['DPD']:,.0f}</td><td class='text-right font-bold text-danger'>{r['ORIGINAL_PRINCIPAL']:,.0f} Đ</td></tr>" for _, r in top_10.iterrows()])
    return f"""<div id="risk" class="section-card"><div class="section-header"><h3>🚩 TOP KHOẢN NỢ TRỌNG ĐIỂM / RỦI RO CAO</h3></div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 25px;"><div><p class="table-title">TOP 10 HỒ SƠ DƯ NỢ LỚN NHẤT (OVERSTANDING)</p><div class="table-container"><table><thead><tr><th>Mã Hồ Sơ</th><th>Dự Án</th><th class="text-right">DPD</th><th class="text-right">Dư nợ gốc</th></tr></thead><tbody>{rows}</tbody></table></div><div style="margin-top: 25px; padding: 15px; background: rgba(239, 68, 68, 0.05); border-radius: 12px; border-left: 4px solid var(--danger);"><p style="color: var(--danger); font-size: 11px; font-weight: 800; margin-bottom: 8px;">💡 CHÚ THÍCH PHÂN TÍCH PARETO (80/20):</p><p style="color: var(--text); font-size: 10.5px; line-height: 1.6; margin: 0;">Biểu đồ Pareto bên cạnh cho thấy mức độ <b>tập trung rủi ro</b>. Nếu đường cong dốc đứng ngay từ đầu, nghĩa là một lượng rất nhỏ hồ sơ (ví dụ 10-20% đầu tiên) đang nắm giữ phần lớn giá trị nợ của hệ thống. <br><br>👉 <b>Chiến lược:</b> Tập trung 80% nguồn lực xử lý vào nhóm "Cá Voi" này sẽ mang lại hiệu quả thu hồi vốn nhanh nhất thay vì dàn trải sang các hồ sơ giá trị thấp.</p></div></div><div style="display: flex; flex-direction: column; gap: 20px;"><div style="background: rgba(239, 68, 68, 0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.1);"><p class="table-title" style="color: var(--danger); text-align: center;">PHÂN TÍCH TỰ TRỌNG PARETO (80/20)</p>{pareto_html}</div><div style="background: rgba(56, 189, 248, 0.03); padding: 15px; border-radius: 12px;"><p class="table-title" style="text-align: center;">TOP 20 HỒ SƠ THEO QUY MÔ DƯ NỢ</p>{bar_html}</div></div></div></div>"""

def render_geo_section(df_hist, df_latest):
    total_cases_all = df_latest['LOAN ID'].nunique()
    
    def get_unique_metric_hist(idx, col):
        return df_hist.loc[idx].drop_duplicates('LOAN ID')[col].sum()

    # Tổng hợp dữ liệu theo Vùng Miền
    geo_agg = df_hist.groupby('PHÂN LOẠI VÙNG MIỀN').agg({
        'LOAN ID': 'nunique',
        'ORIGINAL_PRINCIPAL': lambda x: get_unique_metric_hist(x.index, 'ORIGINAL_PRINCIPAL'),
        'TOTAL_DEBT': lambda x: get_unique_metric_hist(x.index, 'TOTAL_DEBT'),
        'KẾT QUẢ': 'sum',
        'MỤC TIÊU VNE T01': 'sum',
        'MỤC TIÊU ĐỐI TÁC': 'sum'
    }).reset_index().sort_values('KẾT QUẢ', ascending=False)

    def build_geo_rows(data):
        rows = ""
        for _, r in data.iterrows():
            ror = r['KẾT QUẢ'] / r['ORIGINAL_PRINCIPAL'] if r['ORIGINAL_PRINCIPAL'] > 0 else 0
            hs_pct = r['LOAN ID'] / total_cases_all if total_cases_all > 0 else 0
            
            p_target = r['MỤC TIÊU ĐỐI TÁC']
            p_pct_str = f"{r['KẾT QUẢ'] / p_target:.1%}" if p_target > 0 else " - "
            p_color = 'text-success' if (p_target > 0 and r['KẾT QUẢ'] / p_target >= 1) else 'text-danger' if p_target > 0 else "opacity-50"
            
            v_target = r['MỤC TIÊU VNE T01']
            v_pct_str = f"{r['KẾT QUẢ'] / v_target:.1%}" if v_target > 0 else " - "
            v_color = 'text-success' if (v_target > 0 and r['KẾT QUẢ'] / v_target >= 1) else 'text-danger' if v_target > 0 else "opacity-50"
            
            rows += f"<tr><td>{r['PHÂN LOẠI VÙNG MIỀN']}</td><td class='text-right'>{r['LOAN ID']:,}</td><td class='text-right font-bold text-primary'>{hs_pct:.1%}</td><td class='text-right'>{r['ORIGINAL_PRINCIPAL']:,.0f}</td><td class='text-right'>{r['TOTAL_DEBT']:,.0f}</td><td class='text-right' style='color: #fbbf24; font-weight: 800;'>{r['MỤC TIÊU ĐỐI TÁC']:,.0f}</td><td class='text-right text-warn font-bold'>{r['MỤC TIÊU VNE T01']:,.0f}</td><td class='text-right font-bold text-primary'>{r['KẾT QUẢ']:,.0f}</td><td class='text-right text-success font-bold'>{ror:.2%}</td><td class='text-right {p_color} font-bold'>{p_pct_str}</td><td class='text-right {v_color} font-bold'>{v_pct_str}</td></tr>"
        return rows

    geo_rows = build_geo_rows(geo_agg)
    
    # Biểu đồ Donut cho 6 vùng
    fig = px.pie(geo_agg, values='KẾT QUẢ', names='PHÂN LOẠI VÙNG MIỀN', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
    geo_chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    return f"""<div id="geo" class="section-card">
        <div class="section-header"><h3>📍 PHÂN TÍCH THEO KHU VỰC (6 VÙNG)</h3></div>
        <div style="display: flex; flex-direction: column; gap: 30px;">
            <div>
                <p class="table-title">HIỆU SUẤT KHU VỰC CHI TIẾT</p>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr><th>Khu vực</th><th class='text-right'>Hồ sơ</th><th class='text-right'>% HS</th><th class='text-right'>Nợ gốc</th><th class='text-right'>Tổng nợ</th><th class='text-right'>Mục tiêu đối tác</th><th class='text-right'>VNE Target</th><th class='text-right'>Kết quả</th><th class='text-right'>ROR</th><th class='text-right'>% Partner Target</th><th class='text-right'>% VNE Target</th></tr>
                        </thead>
                        <tbody>{geo_rows}</tbody>
                    </table>
                </div>
            </div>
            <div style="background: rgba(56,189,248,0.02); border-radius: 12px; padding: 20px;">
                <p style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: 800; margin-bottom: 20px; letter-spacing: 0.05em; text-align: center;">Tỷ trọng thực thu theo khu vực mới (Hà Nội & HCM tách biệt)</p>
                {geo_chart_html}
            </div>
        </div>
    </div>"""

def build_full_dashboard(df_hist, df_latest, title, role="CEO", active_partner=None):
    sidebar = render_sidebar(); nav = render_nav(active_partner)
    kpis = render_kpi_cards(df_hist, df_latest)
    partner_section = render_partner_section(df_latest) if role == "CEO" else ""
    rev_section = render_revenue_section(df_hist)
    team_section = render_team_performance(df_hist, df_latest)
    trend_section = render_trend_module(df_hist)
    risk_section = render_risk_section(df_latest)
    geo_section = render_geo_section(df_hist, df_latest)
    d_title = "TỔNG QUAN DANH MỤC" if "COMMAND CENTER" in title else title
    
    iframe_html = '<iframe id="vne-risk-iframe" src="Data_Science/Reports/8g_MASTER_DASHBOARD.html" style="width:100%; height:calc(100vh - 42px); border:none; display:none; position:fixed; top:42px; left:0; z-index:1500;" onload="if(typeof syncIframeTheme===\'function\') syncIframeTheme(document.body.classList.contains(\'dark-mode\'))"></iframe>' if not active_partner else ""
    js_funcs = """
        function syncIframeTheme(isDark) {
            try {
                var iframe = document.getElementById('vne-risk-iframe');
                if (iframe && iframe.contentWindow) {
                    iframe.contentWindow.postMessage({ theme: isDark ? 'dark' : 'light' }, '*');
                }
            } catch(e) {}
        }
        
        window.addEventListener('message', function(e) {
            if (e.data && e.data.type === 'request_theme') {
                var isDark = document.body.classList.contains('dark-mode');
                if (e.source) {
                    e.source.postMessage({ theme: isDark ? 'dark' : 'light' }, '*');
                }
            }
        });
        function showHome() {
            document.getElementById('vne-risk-iframe').style.display = 'none';
            document.getElementById('nav-btn-home').classList.add('active');
            document.getElementById('nav-btn-risk').classList.remove('active');
            var partnersNav = document.querySelector('.nav-bar.partners');
            if (partnersNav) partnersNav.style.display = 'flex';
        }
        function showRisk() {
            document.getElementById('vne-risk-iframe').style.display = 'block';
            document.getElementById('nav-btn-home').classList.remove('active');
            document.getElementById('nav-btn-risk').classList.add('active');
            var partnersNav = document.querySelector('.nav-bar.partners');
            if (partnersNav) partnersNav.style.display = 'none';
            
            // Sync theme immediately
            var isDark = document.body.classList.contains('dark-mode');
            syncIframeTheme(isDark);
            
            try {
                var iframe = document.getElementById('vne-risk-iframe');
                if (iframe && iframe.contentWindow) {
                    iframe.contentWindow.dispatchEvent(new Event('resize'));
                }
            } catch(e) {}
        }
        window.addEventListener('DOMContentLoaded', () => {
            if (window.location.search.includes('view=risk')) {
                showRisk();
            }
        });
    """ if not active_partner else ""

    return f"""<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet"><style>
    :root {{
        /* Option 2: Soft Gray / Silver (Light Mode) */
        --bg: #F1F5F9; --card: #F8FAFC; --border: #CBD5E1; --text: #334155; --primary: #0F172A; --muted: #64748B;
        --success: #10B981; --danger: #EF4444; --warn: #F59E0B; --sidebar-w: 210px;
    }}
    body.dark-mode {{
        /* Dark Mode Pairing for Option 2 */
        --bg: #0F172A; --card: #1E293B; --border: #334155; --text: #F1F5F9; --primary: #38BDF8; --muted: #94A3B8;
    }}
    body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); padding:0; margin:0; display:flex; overflow:hidden; transition: background 0.3s, color 0.3s; }}
    .nav-container{{position:fixed;top:0;left:0;right:0;z-index:2000;background:var(--card);backdrop-filter:blur(10px);border-bottom:1px solid var(--border); width: 100%; opacity: 0.98;}}
    .nav-bar{{display:flex;gap:15px;padding:8px 20px;align-items:center}}.nav-bar.partners{{background:rgba(37,99,235,0.03);border-top:1px solid var(--border)}}.nav-link{{text-decoration:none;color:var(--muted);font-size:10px;font-weight:600;padding:5px 8px;border-radius:6px;transition:0.2s}}.nav-link:hover,.nav-link.active{{color:var(--primary);background:var(--bg)}}.brand{{font-weight:800;color:var(--primary);margin-right:20px;font-size:13px}}
    .sidebar{{width:var(--sidebar-w);background:var(--card);backdrop-filter:blur(15px);height:calc(100vh - 84px);position:fixed;left:0;top:84px;border-right:1px solid var(--border);padding:25px 15px;display:flex;flex-direction:column;z-index:1000; box-sizing: border-box; opacity: 0.95;}}
    .sidebar-nav{{flex:1}}.nav-group-label{{font-size:9px;font-weight:800;color:var(--muted);letter-spacing:0.1em;margin:15px 0 8px 0;text-transform:uppercase}}
    .sidebar-link{{display:block;padding:8px 12px;color:var(--muted);text-decoration:none;font-size:11px;font-weight:600;border-radius:6px;transition:0.2s;margin-bottom:4px}}
    .sidebar-link:hover{{background:rgba(37,99,235,0.1);color:var(--primary)}}.sidebar-footer{{padding-top:12px;border-top:1px solid var(--border);color:var(--muted);font-size:8px;text-align:center}}
    .main-wrapper {{ margin-left: var(--sidebar-w); flex: 1; display: flex; flex-direction: column; overflow: hidden; height: 100vh; position: relative; z-index: 10; }}
    .main-content{{margin-top: 84px; padding:35px 50px; max-width:1400px; margin-left: 0; width: 100%; overflow-y: auto; height: calc(100vh - 84px); box-sizing: border-box;}}
    h1{{font-weight:800;color:var(--primary);font-size:2rem;margin:0 0 30px 0; letter-spacing: -0.02em;}}.kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:20px;margin-bottom:35px}}
    .kpi-card{{background:var(--card);padding:15px 5px;border-radius:14px;border-bottom:4px solid var(--primary);text-align:center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); min-height: 110px; display: flex; flex-direction: column; justify-content: center; border: 1px solid var(--border);}}
    .kpi-card .label{{color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;margin-bottom:8px;display:block;white-space:nowrap}}
    .kpi-card .value{{font-size:18px;font-weight:800;display:block;width:100%;word-wrap:break-word;line-height:1.2;color:var(--text)}}
    .lua-chon-label{{color:var(--primary);font-weight:700;margin-bottom:15px;font-size:1.1rem; border-left: 4px solid var(--primary); padding-left: 12px;}}.partner-grid-full{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:40px}}.partner-card{{background:var(--card);padding:22px;border-radius:12px;border:1px solid var(--border);text-decoration:none;transition:0.3s;display:flex;flex-direction:column;gap:15px;box-shadow: 0 1px 3px rgba(0,0,0,0.05)}}.partner-card:hover{{border-color:var(--primary);transform:translateY(-4px);box-shadow:0 12px 20px -8px rgba(37,99,235,0.15)}}.p-header{{display:flex;justify-content:space-between;align-items:center}}.p-name{{font-size:13px;color:var(--text);font-weight:700;letter-spacing:0.02em}}.p-dpd{{font-size:11px;font-weight:700}}.p-row{{display:flex;justify-content:space-between;align-items:baseline}}.p-cases{{font-size:12px;color:var(--muted);font-weight:500}}.p-rate{{font-size:18px;font-weight:800;color:var(--primary)}}.p-bar-bg{{background:var(--bg);height:5px;border-radius:4px;overflow:hidden;margin-top:2px}}.p-bar-fill{{background:linear-gradient(90deg, var(--primary), #6366f1);height:100%}}.section-card{{background:var(--card);border-radius:15px;padding:22px;margin-bottom:25px;border:1px solid var(--border);scroll-margin-top: 100px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: background 0.3s, border 0.3s}}.section-header{{margin-bottom:20px;border-bottom:1px solid var(--border);padding-bottom:12px}}h3{{margin:0; font-size: 1.2rem; font-weight: 700;}}.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:25px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th{{background:rgba(37,99,235,0.05);padding:10px;text-align:left;color:var(--primary);font-weight:700}}td{{padding:10px;border-bottom:1px solid var(--border)}}.text-right{{text-align:right}}.text-success{{color:var(--success)}}.text-warn{{color:var(--warn)}}html{{scroll-behavior: smooth;}} .main-content::-webkit-scrollbar {{ width: 8px; }} .main-content::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 10px; }}</style></head><body><div class="nav-container">{nav}</div>{sidebar}<div class="main-wrapper"><div class="main-content"><h1>{d_title}</h1>{kpis}{partner_section}{rev_section}{team_section}{trend_section}{risk_section}{geo_section}</div></div>
    <script>
        const toggleBtn = document.getElementById('theme-toggle');
        const body = document.body;
        
        function updateCharts(isDark) {{
            const textCol = isDark ? '#F1F5F9' : '#334155';
            const gridCol = isDark ? '#334155' : '#EBF0F8';
            const zeroCol = isDark ? '#475569' : '#CBD5E1';
            const charts = document.querySelectorAll('.plotly-graph-div');
            charts.forEach(chart => {{
                try {{
                    Plotly.relayout(chart, {{
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'font.color': textCol,
                        'title.font.color': textCol,
                        'xaxis.title.font.color': textCol,
                        'yaxis.title.font.color': textCol,
                        'xaxis.tickfont.color': textCol,
                        'yaxis.tickfont.color': textCol,
                        'legend.font.color': textCol,
                        'xaxis.gridcolor': gridCol,
                        'yaxis.gridcolor': gridCol,
                        'xaxis.zerolinecolor': zeroCol,
                        'yaxis.zerolinecolor': zeroCol,
                        'hoverlabel.bgcolor': isDark ? '#1E293B' : '#F8FAFC',
                        'hoverlabel.font.color': textCol
                    }});
                }} catch(e) {{ console.log('Chart not ready', e); }}
            }});
        }}

        toggleBtn.addEventListener('click', () => {{
            body.classList.toggle('dark-mode');
            const isDark = body.classList.contains('dark-mode');
            toggleBtn.innerHTML = isDark ? '☀️ Light' : '🌙 Dark';
            localStorage.setItem('vne_theme', isDark ? 'dark' : 'light');
            updateCharts(isDark);
            if (typeof syncIframeTheme === 'function') syncIframeTheme(isDark);
        }});

        if (localStorage.getItem('vne_theme') === 'dark') {{
            body.classList.add('dark-mode');
            toggleBtn.innerHTML = '☀️ Light';
            setTimeout(() => {{
                updateCharts(true);
                if (typeof syncIframeTheme === 'function') syncIframeTheme(true);
            }}, 800);
        }}
        {js_funcs}
    </script>
    {iframe_html}</body></html>"""

def main():
    df_h, df_l = load_and_engineer_features()
    hub_path = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\STRATEGY_HUB.html'
    hub_html = build_full_dashboard(df_h, df_l, "TỔNG QUAN COMMAND CENTER", role="CEO")
    with open(hub_path, "w", encoding="utf-8") as f: f.write(hub_html)
    print("HUB UPDATED")
    partners = ["ABBANK", "BDI - SHB", "HANMIR - LOTTE", "HANMIR - MIRA", "LOTTE", "MC", "MSB", "SHB", "SVFC"]
    for p in partners:
        p_latest = df_l[df_l['DỰ ÁN'] == p]; p_hist = df_h[df_h['DỰ ÁN'] == p]
        html = build_full_dashboard(p_hist, p_latest, f"Đối tác: {p}", role="LEADER", active_partner=p)
        out_path = os.path.join(OUTPUT_DIR, f"DASHBOARD_{p.replace(' ', '_')}.html")
        with open(out_path, 'w', encoding='utf-8') as f: f.write(html)
        print(f"   + Exported: {p}")

if __name__ == "__main__": main()

# -*- coding: utf-8 -*-
"""
MODULE 8A — PHÂN TÍCH ĐƯỜNG CONG LÃO HÓA NỢ + SMART TRIAGE (DEBT AGING CURVE + AI-ENHANCED TRIAGE)
Phiên bản: 2.0 (2026-05-26) — Tích hợp DS_SEGMENTATION + Feedback Loop từ 8F
Câu hỏi: Ngưỡng DPD nào xác định ranh giới chiến lược thu hồi? Ai cần được leo thang Pháp lý ngay hôm nay?
Biến nguồn (CLEANED.csv + DS_SEGMENTATION_FINAL.csv):
  - DPD, NỢ GỐC (CLEANED)
  - CỤM_HÀNH_VI, PTP_SCORE_PERCENT, HỒ SƠ KHỞI KIỆN, PHÂN LOẠI POS, SỐ_NGÀY_KHÔNG_THANH_TOÁN (DS_SEG)
Feedback Loop: Đọc feedback_thresholds.json từ 8F để điều chỉnh ngưỡng DPD tự động
Output: reports/Data_Science/AGING_CURVE_ANALYSIS.html
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
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
CONFIG_PATH   = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\config_dpd.json'

# ─── DPD TRIAGE TABLE (8 BANDS) ─────────────────────────────────────────────
DPD_BANDS = [
    (0,    89,   "Tiền kỳ (0-89n)",         "Theo dõi sát và nhắc nợ tự động",         "#BBDEFB"),
    (90,   180,  "Khởi phát (90-180n)",        "Tập trung tiếp cận và tương tác sớm",      "#2196F3"),
    (181,  360,  "Trung hạn (181-360n)",     "Tương tác, vận động và hỗ trợ thanh toán",     "#4CAF50"),
    (361,  540,  "Dài hạn 1 (361-540n)",     "Tương tác, xác minh thông tin và đàm phán thanh toán",                     "#8BC34A"),
    (541,  720,  "Dài hạn 2 (541-720n)",     "Tương tác, xác minh thông tin và đàm phán thanh toán",                 "#FFC107"),
    (721,  1080, "Dài hạn 3 (721-1080n)",    "Tương tác, xác minh thông tin và đàm phán thanh toán",               "#FF9800"),
    (1081, 1440, "Kiên trì 1 (1081-1440n)",    "Thương lượng và đàm phán trả nợ 2-3 lần (mang tính dứt điểm)",      "#FF5722"),
    (1441, 1800, "Kiên trì 2 (1441-1800n)",    "Thương lượng và đàm phán, tiền kiểm pháp lý",              "#E91E63"),
    (1801, 9999, "Nhóm Đặc biệt (>1800n)",  "Đàm phán, pháp lý, khởi kiện",            "#9C27B0"),
]

def exponential_decay(x, a, b, c):
    """Hàm suy giảm mũ: y = a * exp(-b*x) + c"""
    return a * np.exp(-b * x) + c

def run():
    print("=" * 60)
    print("MODULE 8A — DEBT AGING CURVE ANALYSIS")
    print("=" * 60)

    # Load config
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = json.load(f)
    write_off_threshold = cfg.get('write_off_threshold', 1800)

    # Load Feedback Loop (nếu có) — điều chỉnh ngưỡng tự động từ 8F
    early_litigation_dpd = 60   # Mặc định
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, encoding='utf-8') as ff:
                fb = json.load(ff)
            early_litigation_dpd = fb.get('early_litigation_dpd', 60)
            triggered_clusters   = fb.get('triggered_clusters', [])
            print(f"   ⚡ FEEDBACK LOOP ACTIVE: early_litigation_dpd = {early_litigation_dpd} ngày")
            if triggered_clusters:
                print(f"   ⚡ Cụm kích hoạt: {triggered_clusters}")
        except Exception as e:
            print(f"   ⚠ Không đọc được feedback_thresholds.json: {e}")
    else:
        print(f"   ℹ Không tìm thấy feedback_thresholds.json — dùng ngưỡng mặc định DPD ≤ {early_litigation_dpd}")

    # ── 1. Load & Prep ───────────────────────────────────
    print("\n[1/4] Đang nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df.get('KẾT QUẢ'), errors='coerce').fillna(0)
    df['DPD']     = pd.to_numeric(df.get('DPD'),      errors='coerce')
    df['NỢ GỐC']  = pd.to_numeric(df.get('NỢ GỐC'),  errors='coerce').fillna(0)

    # ── 2. Collapse Long → Wide per LOAN ID ────────────────────
    print("[2/4] Gom nhóm theo LOAN ID (long → wide)...")
    agg = df.groupby('LOAN ID').agg(
        DPD_CUOI    = ('DPD',      'last'),
        KQ_TONG     = ('KẾT QUẢ', 'sum'),
        NO_GOC      = ('NỢ GỐC',  'last'),
        SO_DONG     = ('LOAN ID',  'count'),
        DU_AN       = ('DỰ ÁN',   'first'),
    ).reset_index()

    # Merge DS_SEGMENTATION để bổ sung AI signals
    print("   Nạp DS_SEGMENTATION_FINAL.csv...")
    seg_cols = ['LOAN ID', 'CỤM_HÀNH_VI', 'PTP_SCORE_PERCENT',
                'HỒ SƠ KHỞI KIỆN', 'PHÂN LOẠI POS', 'SỐ_NGÀY_KHÔNG_THANH_TOÁN', 'CỜ_DI_CƯ']
    # Chỉ lấy cột có trong file
    seg_check = pd.read_csv(DS_SEG_PATH, nrows=1, low_memory=False)
    seg_cols_avail = [c for c in seg_cols if c in seg_check.columns]
    seg = pd.read_csv(DS_SEG_PATH, low_memory=False, usecols=seg_cols_avail)
    seg = seg.drop_duplicates(subset='LOAN ID', keep='last')
    agg = agg.merge(seg, on='LOAN ID', how='left')

    # Chuẩn hóa các cột bổ sung
    if 'PTP_SCORE_PERCENT' in agg.columns:
        agg['PTP_SCORE_PERCENT'] = pd.to_numeric(agg['PTP_SCORE_PERCENT'], errors='coerce').fillna(0)
    if 'CỜ_DI_CƯ' in agg.columns:
        agg['CỜ_DI_CƯ'] = pd.to_numeric(agg['CỜ_DI_CƯ'], errors='coerce').fillna(0)
    if 'HỒ SƠ KHỞI KIỆN' in agg.columns:
        agg['FLAG_LEGAL'] = agg['HỒ SƠ KHỞI KIỆN'].notna().astype(int)
    else:
        agg['FLAG_LEGAL'] = 0

    print(f"   → Tổng số LOAN ID duy nhất (raw): {df['LOAN ID'].nunique():,}")
    print(f"   → Tổng số LOAN ID sau khi gom nhóm: {len(agg):,}")
    
    n_nan_dpd = agg['DPD_CUOI'].isna().sum()
    if n_nan_dpd > 0:
        print(f"   → Lưu ý: Có {n_nan_dpd:,} hồ sơ bị thiếu DPD (NaN).")
        
    agg = agg.dropna(subset=['DPD_CUOI'])
    agg['CÓ_THU_TIỀN'] = (agg['KQ_TONG'] > 0).astype(int)
    total_loans = len(agg)
    print(f"   → Tổng số LOAN ID có DPD hợp lệ: {total_loans:,}")

    # SMART TRIAGE: Những hồ sƠ cần leo thang ngay hôm nay
    if 'PTP_SCORE_PERCENT' in agg.columns and 'CỜ_DI_CƯ' in agg.columns:
        triage_urgent = agg[
            (agg['DPD_CUOI'] <= early_litigation_dpd) &
            (agg['CỜ_DI_CƯ'] == 1) &
            (agg['PTP_SCORE_PERCENT'] < 0.05) &
            (agg['FLAG_LEGAL'] == 0)
        ].copy()
        n_urgent = len(triage_urgent)
        print(f"\n   🚨 SMART TRIAGE: {n_urgent:,} hồ sƠ cần leo thang Pháp lý SớM")
        print(f"      (DPD ≤{early_litigation_dpd}, CỜ_DI_CƯ=1, PTP_SCORE<5%, Chưa có HỒ SƠ KHỞI KIỆN)")
        triage_csv = os.path.join(SUB_DATA_DIR, '8a_triage_urgent.csv')
        triage_urgent[['LOAN ID','DPD_CUOI','NO_GOC','CỤM_HÀNH_VI','PTP_SCORE_PERCENT','CỜ_DI_CƯ','DU_AN']].to_csv(
            triage_csv, index=False, encoding='utf-8-sig')
        print(f"      → Xuất danh sách: {triage_csv}")
    
    n_below_90 = len(agg[agg['DPD_CUOI'] < 90])
    if n_below_90 > 0:
        print(f"   → Lưu ý: Có {n_below_90:,} hồ sơ có DPD < 90 (không nằm trong các dải phân tích).")

    n_above_max = len(agg[agg['DPD_CUOI'] > 9999])
    if n_above_max > 0:
        print(f"   → Lưu ý: Có {n_above_max:,} hồ sơ có DPD > 9999 (nằm ngoài dải tối đa).")

    # ── 3. Tính Recovery Rate theo từng dải DPD ─────────────────
    print("[3/4] Tính tỷ lệ thu hồi theo dải DPD...")

    rows = []
    for lo, hi, label, strategy, color in DPD_BANDS:
        subset = agg[(agg['DPD_CUOI'] >= lo) & (agg['DPD_CUOI'] <= hi)]
        n_total  = len(subset)
        n_paid   = subset['CÓ_THU_TIỀN'].sum()
        rate     = (n_paid / n_total * 100) if n_total > 0 else 0
        avg_dpd  = subset['DPD_CUOI'].mean()
        avg_debt = subset['NO_GOC'].mean()
        rows.append({
            'DPD_MIN':      lo,
            'DPD_MAX':      hi,
            'NHÃN_DẢI':    label,
            'CHIẾN_LƯỢC':  strategy,
            'MÀU':          color,
            'TỔNG_HỒ_SƠ':  n_total,
            'HỒ_SƠ_ĐÃ_TRẢ': n_paid,
            'TỶ_LỆ_THU_HỒI_%': round(rate, 2),
            'DPD_TRUNG_BÌNH':   round(avg_dpd, 0) if not np.isnan(avg_dpd) else 0,
            'NỢ_GỐC_TRUNG_BÌNH': round(avg_debt, 0) if not np.isnan(avg_debt) else 0,
        })

    band_df = pd.DataFrame(rows)
    print(band_df[['NHÃN_DẢI', 'TỔNG_HỒ_SƠ', 'HỒ_SƠ_ĐÃ_TRẢ', 'TỶ_LỆ_THU_HỒI_%']].to_string(index=False))

    # ── 3B. NEW: Tổng tiền thu được & EAD per Band ───────────────
    for lo, hi, label, strategy, color in DPD_BANDS:
        subset = agg[(agg['DPD_CUOI'] >= lo) & (agg['DPD_CUOI'] <= hi)]
        paid_sub = subset[subset['CÓ_THU_TIỀN'] == 1]
        band_df.loc[band_df['NHÃN_DẢI'] == label, 'TỔNG_EAD'] = subset['NO_GOC'].sum()
        band_df.loc[band_df['NHÃN_DẢI'] == label, 'TỔNG_TIỀN_THU'] = paid_sub['KQ_TONG'].sum()
    band_df['EAD_TY']          = (band_df['TỔNG_EAD'] / 1e9).round(2)
    band_df['TIỀN_THU_TY']     = (band_df['TỔNG_TIỀN_THU'] / 1e9).round(3)
    band_df['VÀO_TIỀN_%']      = (band_df['TIỀN_THU_TY'] / band_df['EAD_TY'] * 100).round(2)

    # ── 3C. NEW: Partner breakdown per DPD Band ─────────────────
    agg['DPD_BAND_LABEL'] = 'Unknown'
    for lo, hi, label, strategy, color in DPD_BANDS:
        mask = (agg['DPD_CUOI'] >= lo) & (agg['DPD_CUOI'] <= hi)
        agg.loc[mask, 'DPD_BAND_LABEL'] = label

    partner_band = agg.groupby(['DU_AN', 'DPD_BAND_LABEL']).agg(
        SO_HS = ('LOAN ID', 'count'),
        RATE  = ('CÓ_THU_TIỀN', 'mean'),
        EAD   = ('NO_GOC', 'sum'),
    ).reset_index()
    partner_band['RATE_%'] = (partner_band['RATE'] * 100).round(2)
    partner_band['EAD_TY'] = (partner_band['EAD'] / 1e9).round(2)

    # Heatmap: Partner × Band → Recovery Rate
    partner_band_pivot = partner_band.pivot_table(
        index='DU_AN', columns='DPD_BAND_LABEL',
        values='RATE_%', fill_value=0
    )
    # Preserve band order
    band_order_cols = [b[2] for b in DPD_BANDS if b[2] in partner_band_pivot.columns]
    partner_band_pivot = partner_band_pivot[band_order_cols]
    print("\n   Recovery Rate (%) Theo Đối Tác × DPD Band:")
    print(partner_band_pivot.round(1).to_string())

    # ── 4. Fit Exponential Decay ─────────────────────────────────
    x_vals = band_df['DPD_TRUNG_BÌNH'].values.astype(float)
    y_vals = band_df['TỶ_LỆ_THU_HỒI_%'].values.astype(float)

    fitted_x = np.linspace(x_vals.min(), x_vals.max(), 300)
    fitted_y = None
    inflection_dpd = None

    try:
        popt, _ = curve_fit(
            exponential_decay, x_vals, y_vals,
            p0=[y_vals.max(), 0.001, y_vals.min()],
            maxfev=10000, bounds=([0, 0, 0], [200, 1, 50])
        )
        fitted_y = exponential_decay(fitted_x, *popt)

        # Tìm điểm gãy: nơi đạo hàm thứ 2 đạt cực đại (curvature cao nhất)
        dy2 = np.gradient(np.gradient(fitted_y))
        peak_idx = np.argmax(np.abs(dy2))
        inflection_dpd = int(fitted_x[peak_idx])
        print(f"\n   → Điểm gãy (inflection) phát hiện tại DPD ≈ {inflection_dpd} ngày")
        print(f"   → Tại đó tỷ lệ thu hồi ≈ {fitted_y[peak_idx]:.2f}%")
    except Exception as e:
        print(f"   ⚠ Không fit được đường cong: {e}")

    # ── 5. Visualization ─────────────────────────────────────────
    print("\n[4/4] Tạo Dashboard HTML cao cấp (v3.0 — Risk Manager Edition)...")
    import plotly.offline as plo

    # Tab 1: Curve & Volume
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Đường Cong Lão Hóa Nợ (Aging Curve)", "Khối Lượng Hồ Sơ Theo Trạng Thái", "Dư Nợ Gốc Bình Quân", ""), specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "bar"}, {"type": "domain"}]], vertical_spacing=0.15, horizontal_spacing=0.1)
    
    # Aging Curve
    fig1.add_trace(go.Bar(x=band_df['NHÃN_DẢI'], y=band_df['TỶ_LỆ_THU_HỒI_%'], marker_color=band_df['MÀU'].tolist(), name='Thực tế (%)', text=[f"{v:.1f}%" for v in band_df['TỶ_LỆ_THU_HỒI_%']], textposition='outside'), row=1, col=1)
    if fitted_y is not None:
        fig1.add_trace(go.Scatter(x=band_df['NHÃN_DẢI'], y=exponential_decay(x_vals, *popt), mode='lines+markers', line=dict(color='#E53935', width=2, dash='dash'), name='Dự báo (Exp Fit)'), row=1, col=1)
    if inflection_dpd:
        closest = band_df.iloc[(band_df['DPD_TRUNG_BÌNH'] - inflection_dpd).abs().argsort()[:1]]
        fig1.add_annotation(x=closest['NHÃN_DẢI'].values[0], y=closest['TỶ_LỆ_THU_HỒI_%'].values[0] + 3, text=f"⚠ Điểm Gãy<br>DPD ≈ {inflection_dpd}", showarrow=True, arrowhead=2, bgcolor="#FFF3E0", bordercolor="#FF6D00", row=1, col=1)

    # Volume
    fig1.add_trace(go.Bar(x=band_df['NHÃN_DẢI'], y=band_df['TỔNG_HỒ_SƠ'], marker_color=band_df['MÀU'].tolist(), text=band_df['TỔNG_HỒ_SƠ'].apply(lambda v: f"{v:,}"), textposition='outside'), row=1, col=2)
    # Avg Debt
    fig1.add_trace(go.Bar(x=band_df['NHÃN_DẢI'], y=band_df['NỢ_GỐC_TRUNG_BÌNH'], marker_color='#78909C', text=band_df['NỢ_GỐC_TRUNG_BÌNH'].apply(lambda v: f"{v/1e6:.1f}M"), textposition='outside'), row=2, col=1)
    
    fig1.update_layout( height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=80, l=40, r=40), showlegend=False)
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Fig 2: Triage Plotly Table
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=["<b>Dải DPD</b>", "<b>Tỷ Lệ Thu Hồi</b>", "<b>Số Hồ Sơ</b>", "<b>EAD (Tỷ)</b>", "<b>Tiền Thu (Tỷ)</b>", "<b>Chiến Lược Khuyến Nghị</b>"],
                    fill_color='#1E88E5', font=dict(color='white', size=12), align='left'),
        cells=dict(values=[
            band_df['NHÃN_DẢI'],
            band_df['TỶ_LỆ_THU_HỒI_%'].apply(lambda v: f"{v:.1f}%"),
            band_df['TỔNG_HỒ_SƠ'].apply(lambda v: f"{v:,}"),
            band_df['EAD_TY'].apply(lambda v: f"{v:.2f} Tỷ"),
            band_df['TIỀN_THU_TY'].apply(lambda v: f"{v:.3f} Tỷ"),
            band_df['CHIẾN_LƯỢC']
        ],
        fill_color=[['#1E293B' if i % 2 == 0 else '#0F172A' for i in range(len(band_df))]],
        font=dict(color='white', size=11), align='left')
    )])
    fig2.update_layout(height=520, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=20, r=20))
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 3: Partner × DPD Band Heatmap ───────────────────
    fig3 = go.Figure(data=go.Heatmap(
        z=partner_band_pivot.values,
        x=partner_band_pivot.columns.tolist(),
        y=partner_band_pivot.index.tolist(),
        colorscale='RdYlGn',
        text=partner_band_pivot.values.round(1),
        texttemplate="%{text}%",
        showscale=True,
        colorbar=dict(title='Recovery Rate (%)')
    ))
    fig3.update_layout(
        title="Heatmap Recovery Rate (%) Theo Đối Tác × DPD Band",
        height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=60, l=40, r=40)
    )
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 4: EAD vs Recovery Value per Band ───────────────
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("EAD (Tỷ VND) Theo DPD Band", "Tiền Thu Được (Tỷ VND) Theo DPD Band"), horizontal_spacing=0.1)
    fig4.add_trace(go.Bar(
        x=band_df['NHÃN_DẢI'], y=band_df['EAD_TY'],
        marker_color=band_df['MÀU'].tolist(),
        text=[f"{v:.2f}T" for v in band_df['EAD_TY']], textposition='outside'
    ), row=1, col=1)
    fig4.add_trace(go.Bar(
        x=band_df['NHÃN_DẢI'], y=band_df['TIỀN_THU_TY'],
        marker_color='#10B981',
        text=[f"{v:.3f}T" for v in band_df['TIỀN_THU_TY']], textposition='outside'
    ), row=1, col=2)
    fig4.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=80, l=40, r=40))
    fig4.update_xaxes(tickangle=-20)
    div4 = plo.plot(fig4, output_type='div', include_plotlyjs=False)

    # ─── HTML Data Table ─────────────────────────────────────────
    band_table_rows = ""
    for _, r in band_df.iterrows():
        rate_color = "#10B981" if r['TỶ_LỆ_THU_HỒI_%'] > 15 else ("#F59E0B" if r['TỶ_LỆ_THU_HỒI_%'] > 5 else "#EF4444")
        band_table_rows += f"""
        <tr>
            <td style="font-weight:600; font-size:11.5px;">{r['NHÃN_DẢI']}</td>
            <td style="text-align:right;">{r['TỔNG_HỒ_SƠ']:,.0f}</td>
            <td style="text-align:right; font-weight:700; color:{rate_color};">{r['TỶ_LỆ_THU_HỒI_%']:.1f}%</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right; color:#10B981;">{r['TIỀN_THU_TY']:.3f} Tỷ</td>
            <td style="text-align:right;">{r['NỢ_GỐC_TRUNG_BÌNH']/1e6:.1f}M</td>
            <td style="font-size:11px; color:var(--muted);">{r['CHIẾN_LƯỢC']}</td>
        </tr>"""


    # Insights HTML
    insights_html = ""
    if inflection_dpd:
        insights_html += f"""<div class="alert-box alert-danger"><strong>🔴 Điểm gãy tỷ lệ thu hồi:</strong> Mô hình phát hiện điểm rơi mạnh tại DPD ≈ {inflection_dpd} ngày. Sau mốc này, hiệu quả thu hồi gần như bằng không. Cần chuyển giao qua hành động pháp lý ngay TRƯỚC mốc này.</div>"""
    else:
        insights_html += f"""<div class="alert-box alert-warn"><strong>🟡 Không có điểm gãy rõ ràng:</strong> Tỷ lệ thu hồi giảm từ từ, xem xét tối ưu từng dải băng theo nguồn lực.</div>"""

    insights_html += f"""
    <div class="alert-box alert-success">
        <strong>🟢 Phân bổ thông minh (SMART TRIAGE):</strong> Lọc nhanh được {n_urgent if 'n_urgent' in locals() else 0:,} hồ sơ "báo động đỏ" (DPD sớm nhưng PTP siêu thấp, cờ di cư = 1). Các hồ sơ này đã được kết xuất ra CSV <code>8a_triage_urgent.csv</code> để xử lý ngay.
    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8A — Debt Aging Analysis</title>
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
.data-table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 0; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
.data-table th {{ background: #0F172A; padding: 10px 14px; font-weight: 700; color: var(--primary); border-bottom: 2px solid var(--border); text-align: left; white-space: nowrap; }}
.data-table td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
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
    <h1>8A — DEBT AGING CURVE & SMART TRIAGE</h1>
    <p>Phân tích hàm suy giảm xác suất thu hồi nợ theo thời gian (DPD) và phân lớp rủi ro thông minh AI — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Hồ Sơ Phân Tích</div>
        <div class="kpi-value">{total_loans:,}</div>
        <div class="kpi-sub">Số hồ sơ có DPD</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Điểm Gãy Thu Hồi</div>
        <div class="kpi-value">{inflection_dpd if inflection_dpd else 'N/A'} DPD</div>
        <div class="kpi-sub">Ngưỡng giới hạn cần leo thang trước</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Dự Báo Lệnh Di Cư</div>
        <div class="kpi-value">{n_urgent if 'n_urgent' in locals() else 0:,} HS</div>
        <div class="kpi-sub">Cần khoá hồ sơ khẩn cấp hôm nay</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">ARR Feedback Threshold</div>
        <div class="kpi-value">{early_litigation_dpd} DPD</div>
        <div class="kpi-sub">Mức DPD khởi kiện điều chỉnh tự động</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Đường Cong Lão Hóa Nợ</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. EAD & Tiền Thu Theo Band 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. Đối Tác × DPD Band 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Bảng Triage Chi Tiết 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Nhận Định & Kế Hoạch</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="chart-card">{div1}</div>
</div>

<div id="tab2" class="tab-content">
    <div class="chart-card">{div4}</div>
</div>

<div id="tab3" class="tab-content">
    <div class="chart-card">{div3}</div>
</div>

<div id="tab4" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 Bảng Chi Tiết Triage Theo DPD Band + EAD + Tiền Thu</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>DPD Band</th><th>Số Hồ Sơ</th><th>Recovery Rate</th>
                <th>EAD (Tỷ)</th><th>Tiền Thu (Tỷ)</th><th>Nợ Gốc TB</th><th>Chiến Lược</th>
            </tr></thead>
            <tbody>{band_table_rows}</tbody>
        </table></div>
    </div>
    <div class="chart-card" style="margin-top:20px;">{div2}</div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Chiến Lược Chuyển Dịch Tài Sản Đóng Băng</h3>
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

    out_path = os.path.join(REPORT_DIR, "8a_AGING_CURVE_ANALYSIS.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # ── CSV export ────────────────────────────────────────────────
    csv_path = os.path.join(SUB_DATA_DIR, "8a_aging_triage_table.csv")
    band_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8A!")
    print(f"   → HTML: {out_path}")
    print(f"   → CSV:  {csv_path}")
    if inflection_dpd:
        print(f"\n📌 KEY INSIGHT: Điểm gãy tỷ lệ thu hồi ≈ DPD {inflection_dpd} ngày")
        print(f"   Đây là mốc quan trọng nhất để phân chia chiến lược thu hồi.")

if __name__ == "__main__":
    run()
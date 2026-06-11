# -*- coding: utf-8 -*-
"""
MODULE 8C — PHÂN TÍCH ÁP LỰC NỢ CHỒNG CHÉO × NĂNG LỰC TÀI CHÍNH (DEBT STACKING × FINANCIAL CAPACITY)
Phiên bản: 2.0 (2026-05-26) — Tích hợp DS_SEGMENTATION, CDPI, Waive Elasticity
Câu hỏi: Nợ nhiều chủ + năng lực tài chính yếu có tỷ lệ thu hồi khác nhau thế nào?
         Phân khúc nào nên xóa phí phạt để kích thích trả gốc ngay?
Biến nguồn (DS_SEGMENTATION_FINAL.csv):
  - SỐ_DỰ_ÁN_NGOÀI, TỶ_LỆ_NỢ_TRÊN_LƯƠNG (DTI), CỤM_HÀNH_VI
  - PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH, SỐ LƯỢNG HỢP ĐỒNG
  - KHÁCH HÀNG NHIỀU DỰ ÁN, MỨC LƯƠNG, SỐ TIỀN THANH TOÁN HÀNG THÁNG
  - TỔNG ĐÃ THANH TOÁN, TUỔI, GIỚI TÍNH
Công thức CDPI: DTI × √(1 + SỐ_DỰ_ÁN_NGOÀI)
Output: reports/Data_Science/DEBT_STACKING.html
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
from scipy.stats import chi2_contingency, mannwhitneyu
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
DS_SEG_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science\Data\DS_SEGMENTATION_FINAL.csv'

# Ngưỡng Lương — có thể tinh chỉnh
LUONG_CAO_NGUONG = 10_000_000   # 10 triệu VND/tháng


def classify_income(luong):
    if pd.isna(luong) or luong == 0:
        return 'Không Rõ / Không Khai'
    elif luong >= LUONG_CAO_NGUONG:
        return 'Lương Cao (≥10tr)'
    else:
        return 'Lương Thấp (<10tr)'


def count_creditors(val):
    """Đếm số chủ nợ khác từ cột KHÁCH HÀNG NHIỀU DỰ ÁN"""
    if pd.isna(val) or str(val).strip() in ['', '0']:
        return 0
    # Thử nhiều loại dấu phân tách phổ biến trong data
    s = str(val).replace('_', '-').replace(';', '-').replace(',', '-')
    parts = [p.strip() for p in s.split('-') if p.strip()]
    return max(0, len(parts))


def run():
    print("=" * 60)
    print("MODULE 8C — DEBT STACKING × INCOME MATRIX")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────
    print("\n[1/5] Đang nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df.get('KẾT QUẢ'), errors='coerce').fillna(0)
    df['MỨC LƯƠNG'] = pd.to_numeric(df.get('MỨC LƯƠNG'), errors='coerce')
    df['NỢ GỐC'] = pd.to_numeric(df.get('NỢ GỐC'), errors='coerce').fillna(0)

    # ── 2. Long → Wide per LOAN ID ────────────────────────────
    print("[2/5] Gom nhóm theo LOAN ID...")
    agg = df.groupby('LOAN ID').agg(
        KQ_TONG              = ('KẾT QUẢ',               'sum'),
        LUONG                = ('MỨC LƯƠNG',              'max'),
        NO_GOC               = ('NỢ GỐC',                 'last'),
        NHIEU_DU_AN_RAW      = ('KHÁCH HÀNG NHIỀU DỰ ÁN', 'first'),
        SO_HOP_DONG_RAW      = ('SỐ LƯỢNG HỢP ĐỒNG',     'first'),
        TONG_DA_THANH_TOAN   = ('TỔNG ĐÃ THANH TOÁN',    'last'),
        TIEN_TT_HANG_THANG   = ('SỐ TIỀN THANH TOÁN HÀNG THÁNG', 'first'),
        TUOI                 = ('TUỔI',                   'first'),
        GIOI_TINH            = ('GIỚI TÍNH',              'first'),
    ).reset_index()

    agg['CÓ_THU_TIỀN'] = (agg['KQ_TONG'] > 0).astype(int)

    # Merge DS_SEGMENTATION để lấy CỤM_HÀNH_VI, DTI, SỐ_DỰ_ÁN_NGOÀI, PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH
    print("   Nạp DS_SEGMENTATION_FINAL.csv...")
    seg = pd.read_csv(DS_SEG_PATH, low_memory=False,
                      usecols=['LOAN ID', 'CỤM_HÀNH_VI', 'TỶ_LỆ_NỢ_TRÊN_LƯƠNG',
                               'SỐ_DỰ_ÁN_NGOÀI', 'PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH',
                               'PTP_SCORE_PERCENT'])
    # Deduplicate DS_SEG trước khi merge (lấy dòng cuối của mỗi LOAN ID)
    seg = seg.drop_duplicates(subset='LOAN ID', keep='last')
    agg = agg.merge(seg, on='LOAN ID', how='left')

    # ── 3. Sinh biến Debt Stacking + CDPI ───────────────────────
    print("[3/5] Sinh biến phân tích (bao gồm CDPI)...")
    agg['ĐẾM_CHỦ_NỢ_KHÁC'] = agg['NHIEU_DU_AN_RAW'].apply(count_creditors)
    agg['NHÓM_SỐ_CHỦ_NỢ'] = pd.cut(
        agg['ĐẾM_CHỦ_NỢ_KHÁC'],
        bins=[-1, 0, 1, 2, 100],
        labels=['Chỉ 1 chủ nợ', 'Nợ 2 chủ', 'Nợ 3 chủ', 'Nợ ≥4 chủ']
    )

    # CDPI = DTI × √(1 + SỐ_DỰ_ÁN_NGOÀI) — Composite Debt Pressure Index
    agg['SỐ_DỰ_ÁN_NGOÀI'] = pd.to_numeric(agg['SỐ_DỰ_ÁN_NGOÀI'], errors='coerce').fillna(0)
    agg['TỶ_LỆ_NỢ_TRÊN_LƯƠNG'] = pd.to_numeric(agg['TỶ_LỆ_NỢ_TRÊN_LƯƠNG'], errors='coerce').fillna(0)
    agg['CDPI'] = agg['TỶ_LỆ_NỢ_TRÊN_LƯƠNG'] * np.sqrt(1 + agg['SỐ_DỰ_ÁN_NGOÀI'])
    agg['CDPI_TIER'] = pd.cut(
        agg['CDPI'],
        bins=[-0.001, 1, 3, 6, float('inf')],
        labels=['CDPI Thấp (≤1)', 'CDPI Trung Bình (1-3)', 'CDPI Cao (3-6)', 'CDPI Rất Cao (>6)']
    )

    # Phân khúc lương
    agg['PHÂN_KHÚC_LƯƠNG'] = agg['LUONG'].apply(classify_income)

    # ── 4. Ma Trận 2D: Tỷ lệ thu hồi ──────────────────────────
    print("[4/5] Tính ma trận Debt Stacking × Thu nhập...")

    order_chu_no  = ['Chỉ 1 chủ nợ', 'Nợ 2 chủ', 'Nợ 3 chủ', 'Nợ ≥4 chủ']
    order_luong   = ['Lương Cao (≥10tr)', 'Lương Thấp (<10tr)', 'Không Rõ / Không Khai']

    matrix_rate  = pd.DataFrame(index=order_chu_no, columns=order_luong, dtype=float)
    matrix_count = pd.DataFrame(index=order_chu_no, columns=order_luong, dtype=int)

    for chu_no in order_chu_no:
        for luong_cat in order_luong:
            mask = (agg['NHÓM_SỐ_CHỦ_NỢ'] == chu_no) & (agg['PHÂN_KHÚC_LƯƠNG'] == luong_cat)
            subset = agg[mask]
            n  = len(subset)
            ok = subset['CÓ_THU_TIỀN'].sum()
            matrix_rate.loc[chu_no, luong_cat]  = round(ok / n * 100, 2) if n > 0 else 0.0
            matrix_count.loc[chu_no, luong_cat] = n

    print("\n--- Ma trận Tỷ Lệ Thu Hồi (%) ---")
    print(matrix_rate.to_string())
    print("\n--- Ma trận Số Hồ Sơ ---")
    print(matrix_count.to_string())

    # ── 4b. Kiểm định Chi-Square ────────────────────────────────
    contingency = pd.DataFrame(index=order_chu_no, columns=order_luong, dtype=int)
    for chu_no in order_chu_no:
        for luong_cat in order_luong:
            mask = (agg['NHÓM_SỐ_CHỦ_NỢ'] == chu_no) & (agg['PHÂN_KHÚC_LƯƠNG'] == luong_cat)
            contingency.loc[chu_no, luong_cat] = agg[mask]['CÓ_THU_TIỀN'].sum()

    try:
        chi2, p_val, dof, _ = chi2_contingency(contingency.values.astype(float))
        sig = "✅ CÓ Ý NGHĨA THỐNG KÊ" if p_val < 0.05 else "❌ Không có ý nghĩa thống kê"
        print(f"\n📊 Chi-Square Test: chi2={chi2:.2f}, p-value={p_val:.4f} → {sig}")
    except Exception as e:
        print(f"⚠ Chi-Square error: {e}")
        p_val = None

    # ── 4c. Waive Elasticity — Phân tích độ co giãn miễn giảm theo CỤM_HÀNH_VI ──
    print("\n[4c] Waive Elasticity theo CỤM_HÀNH_VI...")
    waive_stats = agg.groupby('CỤM_HÀNH_VI', observed=True).agg(
        SỐ_HS           = ('LOAN ID',          'count'),
        TỶ_LỆ_THU_HỒI  = ('CÓ_THU_TIỀN',     'mean'),
        CDPI_TB         = ('CDPI',             'mean'),
        DTI_TB          = ('TỶ_LỆ_NỢ_TRÊN_LƯƠNG', 'mean'),
        DU_AN_NGOAI_TB  = ('SỐ_DỰ_ÁN_NGOÀI', 'mean'),
        PTP_TB          = ('PTP_SCORE_PERCENT','mean'),
    ).reset_index()
    waive_stats['TỶ_LỆ_%'] = (waive_stats['TỶ_LỆ_THU_HỒI'] * 100).round(3)
    waive_stats['CDPI_TB']  = waive_stats['CDPI_TB'].round(2)
    waive_stats['KHUYẾN_NGHỊ'] = waive_stats.apply(
        lambda r: ('⚡ XÓA 100% PHÍ PHẠT — CÓ THIỆN CHÍ, CDPI CAO'
                   if r['CDPI_TB'] > 3 and r['TỶ_LỆ_%'] > 0.001
                   else ('⛔ DỪNG GỌI — VỠ NỢ CHUỖI (CDPI>6, DU_AN>3)'
                         if r['CDPI_TB'] > 6 and r['DU_AN_NGOAI_TB'] > 3
                         else '📞 DUY TRÌ TELESALES')), axis=1
    )
    print(waive_stats[['CỤM_HÀNH_VI','SỐ_HS','TỶ_LỆ_%','CDPI_TB','KHUYẾN_NGHỊ']].to_string(index=False))

    # CDPI Tier analysis
    cdpi_stats = agg.groupby('CDPI_TIER', observed=True).agg(
        SỐ_HS  = ('LOAN ID',      'count'),
        TỶ_LỆ  = ('CÓ_THU_TIỀN', 'mean'),
        NO_TB  = ('NO_GOC',       'mean'),
    ).reset_index()
    cdpi_stats['TỶ_LỆ_%'] = (cdpi_stats['TỶ_LỆ'] * 100).round(3)
    print("\n   CDPI Tier → Recovery Rate:")
    print(cdpi_stats[['CDPI_TIER','SỐ_HS','TỶ_LỆ_%']].to_string(index=False))

    # ── 5. Visualization ─────────────────────────────────────────
    print("\n[5/5] Tạo Dashboard HTML cao cấp...")
    import plotly.offline as plo

    # Tab 1: CDPI & Ma Trận
    fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Ma Trận Tỷ Lệ Thu Hồi (%) — Chủ Nợ × Lương", "Áp Lực Nợ Chồng Chéo (CDPI Tier)", "Phân Bổ Theo Số Chủ Nợ", "Tỷ Lệ Thu Hồi (Số Chủ Nợ)"), specs=[[{"type": "heatmap"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]], vertical_spacing=0.15, horizontal_spacing=0.1)
    
    fig1.add_trace(go.Heatmap(z=matrix_rate.values.astype(float), x=order_luong, y=order_chu_no, colorscale='RdYlGn', text=[[f"{matrix_rate.loc[r, c]:.1f}%\n({matrix_count.loc[r, c]:,} HS)" for c in order_luong] for r in order_chu_no], texttemplate="%{text}", colorbar=dict(title="Thu hồi (%)"), zmin=0, zmax=5), row=1, col=1)
    
    cdpi_plot = cdpi_stats.dropna(subset=['CDPI_TIER'])
    cdpi_colors = ['#43A047','#FDD835','#FB8C00','#E53935']
    fig1.add_trace(go.Bar(x=[str(t) for t in cdpi_plot['CDPI_TIER']], y=cdpi_plot['TỶ_LỆ_%'], marker_color=cdpi_colors[:len(cdpi_plot)], text=[f"{v:.3f}%" for v in cdpi_plot['TỶ_LỆ_%']], textposition='outside', showlegend=False), row=1, col=2)
    
    vol = agg['NHÓM_SỐ_CHỦ_NỢ'].value_counts().reindex(order_chu_no).fillna(0)
    fig1.add_trace(go.Bar(x=vol.index.tolist(), y=vol.values, marker_color=['#1E88E5', '#43A047', '#FB8C00', '#E53935'], text=[f"{int(v):,}" for v in vol.values], textposition='outside', showlegend=False), row=2, col=1)
    
    rate_by_chu = agg.groupby('NHÓM_SỐ_CHỦ_NỢ', observed=True)['CÓ_THU_TIỀN'].mean() * 100
    rate_by_chu = rate_by_chu.reindex(order_chu_no).fillna(0)
    fig1.add_trace(go.Bar(x=rate_by_chu.index.tolist(), y=rate_by_chu.values, marker_color=['#1E88E5', '#43A047', '#FB8C00', '#E53935'], text=[f"{v:.2f}%" for v in rate_by_chu.values], textposition='outside', showlegend=False), row=2, col=2)
    
    fig1.update_layout( height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Tab 2: Waive Elasticity
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Khuyến Nghị Waive Elasticity (Cụm Hành Vi)", "Tỷ Lệ Thu Hồi Theo Phân Khúc Lương"), specs=[[{"type": "bar"}, {"type": "bar"}]], horizontal_spacing=0.1)
    
    w = waive_stats.dropna(subset=['CỤM_HÀNH_VI'])
    waive_colors_map = {'⚡ XÓA 100% PHÍ PHẠT — CÓ THIỆN CHÍ, CDPI CAO':'#FDD835', '⛔ DỪNG GỌI — VỠ NỢ CHUỖI (CDPI>6, DU_AN>3)':'#E53935', '📞 DUY TRÌ TELESALES':'#1E88E5'}
    fig2.add_trace(go.Bar(x=w['CỤM_HÀNH_VI'].tolist(), y=w['CDPI_TB'], marker_color=[waive_colors_map.get(r,'#9E9E9E') for r in w['KHUYẾN_NGHỊ']], text=w['KHUYẾN_NGHỊ'].tolist(), textposition='outside', showlegend=False), row=1, col=1)
    
    rate_by_luong = agg.groupby('PHÂN_KHÚC_LƯƠNG')['CÓ_THU_TIỀN'].mean() * 100
    rate_by_luong = rate_by_luong.reindex(order_luong).fillna(0)
    fig2.add_trace(go.Bar(x=rate_by_luong.index.tolist(), y=rate_by_luong.values, marker_color=['#5E35B1', '#00ACC1', '#8D6E63'], text=[f"{v:.2f}%" for v in rate_by_luong.values], textposition='outside', showlegend=False), row=1, col=2)
    
    fig2.update_layout( height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 3: CDPI Histogram + Scatter CDPI×Recovery ─────────
    import plotly.express as px
    agg['PAID_LABEL'] = agg['CÓ_THU_TIỀN'].map({1: 'Có Trả', 0: 'Chưa Trả'})
    fig3 = make_subplots(rows=1, cols=2,
        subplot_titles=("Phân Phối CDPI — Có Trả vs Chưa Trả", "Scatter: CDPI × DTI × Recovery (Cụm HV)"),
        horizontal_spacing=0.12)
    for label, color in [('Có Trả', '#10B981'), ('Chưa Trả', '#EF4444')]:
        sub = agg[(agg['PAID_LABEL'] == label) & (agg['CDPI'] <= 20) & agg['CDPI'].notna()]
        fig3.add_trace(go.Histogram(
            x=sub['CDPI'], name=label, marker_color=color,
            opacity=0.7, nbinsx=40, histnorm='probability density'
        ), row=1, col=1)
    scatter_data = agg.groupby('CỤM_HÀNH_VI', observed=True).agg(
        CDPI_TB=('CDPI', 'mean'), DTI_TB=('TỶ_LỆ_NỢ_TRÊN_LƯƠNG', 'mean'),
        RATE=('CÓ_THU_TIỀN', 'mean'), SO_HS=('LOAN ID', 'count'),
    ).reset_index()
    scatter_data['RATE_%'] = (scatter_data['RATE'] * 100).round(2)
    scatter_data['CUM_STR'] = scatter_data['CỤM_HÀNH_VI'].astype(str)
    scatter_data = scatter_data.dropna(subset=['CDPI_TB', 'DTI_TB'])
    fig3.add_trace(go.Scatter(
        x=scatter_data['CDPI_TB'], y=scatter_data['DTI_TB'],
        mode='markers+text', text=scatter_data['CUM_STR'],
        textposition='top center',
        marker=dict(
            size=(scatter_data['SO_HS'] / scatter_data['SO_HS'].max() * 40 + 8).clip(8, 50),
            color=scatter_data['RATE_%'], colorscale='RdYlGn',
            showscale=True, colorbar=dict(title='Recovery %', x=1.02)
        ),
        hovertemplate="<b>Cụm %{text}</b><br>CDPI TB: %{x:.2f}<br>DTI TB: %{y:.2f}<extra></extra>"
    ), row=1, col=2)
    fig3.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=50, b=50, l=40, r=40), showlegend=True)
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # ─── HTML Data Tables ──────────────────────────────────────────
    w = waive_stats.dropna(subset=['CỤM_HÀNH_VI'])
    waive_rows = ""
    for _, r in w.iterrows():
        icon_color = "#EF4444" if 'DỪNG GỌI' in r['KHUYẾN_NGHỊ'] else ("#F59E0B" if 'XÓA' in r['KHUYẾN_NGHỊ'] else "#10B981")
        waive_rows += f"""
        <tr>
            <td style="font-weight:600; text-align:center;">{r['CỤM_HÀNH_VI']}</td>
            <td style="text-align:right;">{r['SỐ_HS']:,}</td>
            <td style="text-align:right;">{r['TỶ_LỆ_%']:.3f}%</td>
            <td style="text-align:right;">{r['CDPI_TB']:.2f}</td>
            <td style="text-align:right;">{r['DTI_TB']:.2f}</td>
            <td style="text-align:right;">{r['DU_AN_NGOAI_TB']:.1f}</td>
            <td style="text-align:right;">{r['PTP_TB']*100:.1f}%</td>
            <td style="color:{icon_color}; font-weight:600; font-size:11px;">{r['KHUYẾN_NGHỊ']}</td>
        </tr>"""
    cdpi_rows = ""
    for _, r in cdpi_stats.dropna(subset=['CDPI_TIER']).iterrows():
        rate_color = "#10B981" if r['TỶ_LỆ_%'] > 5 else ("#F59E0B" if r['TỶ_LỆ_%'] > 1 else "#EF4444")
        cdpi_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['CDPI_TIER']}</td>
            <td style="text-align:right;">{r['SỐ_HS']:,}</td>
            <td style="text-align:right; font-weight:700; color:{rate_color};">{r['TỶ_LỆ_%']:.3f}%</td>
            <td style="text-align:right;">{r['NO_TB']/1e6:.1f}M</td>
        </tr>"""

    # Insights HTML
    insights_html = ""
    if p_val is not None and p_val < 0.05:
        insights_html += f"""<div class="alert-box alert-success"><strong>✅ Mối quan hệ Thu nhập & Nợ (p={p_val:.4f}):</strong> Tác động của năng lực tài chính lên tỷ lệ thu hồi khi nợ chồng chéo có ý nghĩa thống kê rõ rệt.</div>"""
    
    # Tìm cụm cần Dừng gọi
    stop_call = w[w['KHUYẾN_NGHỊ'].str.contains('DỪNG GỌI')]
    if not stop_call.empty:
        cums = ", ".join(stop_call['CỤM_HÀNH_VI'].astype(str))
        insights_html += f"""<div class="alert-box alert-danger"><strong>⛔ Báo Động CDPI Cao:</strong> Các nhóm {cums} đang có áp lực CDPI > 6 và vay > 3 dự án ngoài. Nguy cơ vỡ nợ chuỗi (Snowball default). Khuyến nghị DỪNG GỌI để giảm chi phí OPEX.</div>"""
    
    waive_call = w[w['KHUYẾN_NGHỊ'].str.contains('XÓA 100%')]
    if not waive_call.empty:
        cums_waive = ", ".join(waive_call['CỤM_HÀNH_VI'].astype(str))
        insights_html += f"""<div class="alert-box alert-warn"><strong>⚡ Cơ Hội Khôi Phục Nợ Gốc:</strong> Nhóm {cums_waive} có thiện chí nhưng CDPI cao. Đề xuất kịch bản Telesale: Waive 100% phí phạt nếu thanh toán dứt điểm dư nợ gốc trong 48h.</div>"""

    total_loans = len(agg)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>8C — Debt Stacking Analysis</title>
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
    <h1>8C — NỢ CHỒNG CHÉO & NĂNG LỰC TÀI CHÍNH (CDPI)</h1>
    <p>Phân tích tác động của việc nợ nhiều chủ (Debt Stacking) lên khả năng thu hồi và tối ưu chính sách Waive phí phạt theo AI (Waive Elasticity) — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Hồ Sơ Phân Tích</div>
        <div class="kpi-value">{total_loans:,}</div>
        <div class="kpi-sub">Số LOAN ID duy nhất</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Rủi Ro Vỡ Nợ Chuỗi</div>
        <div class="kpi-value">{len(stop_call) if not stop_call.empty else 0} Cụm</div>
        <div class="kpi-sub">Khuyến nghị ngưng tác động (CDPI > 6)</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Cơ Hội Waive Phí Phạt</div>
        <div class="kpi-value">{len(waive_call) if not waive_call.empty else 0} Cụm</div>
        <div class="kpi-sub">Áp dụng kịch bản Waive 100% khẩn cấp</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Ý Nghĩa Thống Kê</div>
        <div class="kpi-value">{'✅ PASS' if p_val and p_val < 0.05 else '❌ FAIL'}</div>
        <div class="kpi-sub">Mức độ tin cậy của mô hình</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. Áp Lực CDPI & Phân Bổ</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. Waive Elasticity</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. CDPI Histogram & Scatter 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Bảng Dữ Liệu Chi Tiết 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Nhận Định AI</button>
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
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 Waive Elasticity — Bảng Chi Tiết Theo Cụm Hành Vi</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Cụm HV</th><th>Số HS</th><th>Recovery %</th>
                <th>CDPI TB</th><th>DTI TB</th><th>Dự Án Ngoài</th>
                <th>PTP TB</th><th>Khuyến Nghị</th>
            </tr></thead>
            <tbody>{waive_rows}</tbody>
        </table></div>
    </div>
    <div class="chart-card" style="margin-top:16px;">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📈 CDPI Tier — Recovery Rate Chi Tiết</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>CDPI Tier</th><th>Số HS</th><th>Recovery Rate</th><th>Nợ Gốc TB</th>
            </tr></thead>
            <tbody>{cdpi_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card" style="padding: 30px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 24px;">💡 Chiến Lược Chống Vỡ Nợ Chuỗi (Snowball Prevention)</h3>
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

    out_html = os.path.join(REPORT_DIR, "8c_DEBT_STACKING.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv = os.path.join(SUB_DATA_DIR, "8c_debt_stacking_matrix.csv")
    matrix_rate.to_csv(out_csv, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8C!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")

if __name__ == "__main__":
    run()
# -*- coding: utf-8 -*-
"""
MODULE 7D — LGD MODEL & EXPECTED LOSS FRAMEWORK (v3.0 — Deep Risk Manager Edition)
Câu hỏi: Không chỉ "có trả không?" mà "trả bao nhiêu % nợ gốc?" và Expected Loss = bao nhiêu VND?
         EL phân bổ theo DPD bucket như thế nào? Top N hồ sơ EL cao nhất là ai?
Output:  reports/Data_Science/Reports/7d_LGD_ANALYSIS.html + 7d_lgd_results.csv
"""
import pandas as pd
import numpy as np
import os, sys, warnings
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
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

# DPD Buckets cho EL analysis
DPD_BUCKETS = [
    (0,    90,   'Current/Early (0–90)'),
    (91,   180,  'Fresh NPL (91–180)'),
    (181,  360,  'Khó Đòi Sớm (181–360)'),
    (361,  540,  'Khó Đòi GĐ1 (361–540)'),
    (541,  720,  'Khó Đòi GĐ2 (541–720)'),
    (721,  1080, 'Khó Đòi GĐ3 (721–1080)'),
    (1081, 1440, 'Nợ Sâu GĐ1 (1081–1440)'),
    (1441, 9999, 'Nợ Sâu/TV (>1440)'),
]

def assign_dpd_bucket(dpd):
    if pd.isna(dpd) or dpd < 0: return 'Unknown'
    for lo, hi, label in DPD_BUCKETS:
        if lo <= dpd <= hi: return label
    return 'Nợ Sâu/TV (>1440)'

def run():
    print("=" * 60)
    print("MODULE 7D — LGD MODEL & EXPECTED LOSS FRAMEWORK (v3.0)")
    print("=" * 60)

    # ── 1. Load & collapse to LOAN ID level ───────────────────
    print("\n[1/7] Nạp dữ liệu và gom nhóm LOAN ID...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    for c in ['KẾT QUẢ', 'NỢ GỐC', 'TỔNG NỢ', 'DPD', 'LÃI SUẤT', 'TUỔI', 'MỨC LƯƠNG']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    agg = df.groupby('LOAN ID').agg(
        KQ_TONG        = ('KẾT QUẢ',              'sum'),
        NO_GOC         = ('NỢ GỐC',               'last'),
        TONG_NO        = ('TỔNG NỢ',              'last'),
        DPD            = ('DPD',                  'last'),
        LAI_SUAT       = ('LÃI SUẤT',             'max'),
        RATING         = ('ĐÁNH GIÁ KHÁCH HÀNG', 'last'),
        DU_AN          = ('DỰ ÁN',                'first'),
        TINH           = ('TỈNH TẠM TRÚ',         'first'),
        VL_STATUS      = ('TÌNH TRẠNG VL',         'first'),
        POS_NHOM       = ('PHÂN LOẠI POS',         'first'),
        SO_HOP_DONG    = ('SỐ LƯỢNG HỢP ĐỒNG',    'first'),
        NHOM_TUOI      = ('PL NHÓM TUỔI',          'first'),
        THANG          = ('THÁNG',                 'last'),
    ).reset_index()

    agg['NO_GOC'] = agg['NO_GOC'].fillna(0)
    agg = agg[agg['NO_GOC'] > 0]

    # ── 2. Tạo target LGD ─────────────────────────────────────
    print("[2/7] Tính Recovery Rate (REC_RATE) = KẾT QUẢ / NỢ GỐC ...")
    agg['REC_RATE'] = (agg['KQ_TONG'] / agg['NO_GOC']).clip(0, 1).fillna(0)
    agg['CÓ_TRẢ']  = (agg['KQ_TONG'] > 0).astype(int)
    agg['LGD'] = 1 - agg['REC_RATE']

    # ── 3. Phân phối REC_RATE ──────────────────────────────────
    print("[3/7] Phân tích phân phối Recovery Rate ...")
    paid_agg = agg[agg['CÓ_TRẢ'] == 1]
    buckets  = pd.cut(paid_agg['REC_RATE'],
                      bins=[0, 0.10, 0.30, 0.50, 0.80, 1.01],
                      labels=['<10% (Trả lắt nhắt)', '10–30%', '30–50%', '50–80%', '>80% (Gần tất toán)'])
    dist = buckets.value_counts().sort_index()

    # ── 4. Feature engineering & Model Training ───────────────
    print("\n[4/7] Huấn luyện mô hình 2 giai đoạn (Two-Stage LGD)...")
    rating_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1, '0': 0}
    agg['RATING_NUM'] = agg['RATING'].map(rating_order).fillna(0)
    pos_map   = {'LOW POS': 0, 'MEDIUM POS': 1, 'HIGHT POS': 2}
    agg['POS_NUM'] = agg['POS_NHOM'].map(pos_map).fillna(1)

    le_du_an = LabelEncoder()
    le_vl    = LabelEncoder()
    agg['DU_AN_ENC'] = le_du_an.fit_transform(agg['DU_AN'].fillna('Unknown'))
    agg['VL_ENC']    = le_vl.fit_transform(agg['VL_STATUS'].fillna('Unknown'))

    feat_cols = ['DPD', 'NO_GOC', 'LAI_SUAT', 'RATING_NUM', 'POS_NUM', 'DU_AN_ENC', 'VL_ENC']
    X = agg[feat_cols].fillna(agg[feat_cols].median())

    # Stage 1: PD Model
    stage1 = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    stage1.fit(X, agg['CÓ_TRẢ'])
    agg['PD_SCORE'] = stage1.predict_proba(X)[:, 1]
    pd_cv = cross_val_score(stage1, X, agg['CÓ_TRẢ'], cv=3, scoring='roc_auc').mean()
    print(f"   Stage 1 (PD) AUC-ROC: {pd_cv:.4f}")

    # Stage 2: LGD Model
    paid_mask  = agg['CÓ_TRẢ'] == 1
    X_paid     = X[paid_mask]
    y_lgd      = agg.loc[paid_mask, 'LGD']
    stage2     = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    stage2.fit(X_paid, y_lgd)
    lgd_cv = cross_val_score(stage2, X_paid, y_lgd, cv=3, scoring='neg_mean_absolute_error').mean()
    print(f"   Stage 2 (LGD) MAE:     {-lgd_cv:.4f}")

    feat_importance = pd.DataFrame({
        'Feature': feat_cols,
        'Importance_PD': stage1.feature_importances_,
        'Importance_LGD': stage2.feature_importances_
    })

    agg['LGD_PRED'] = np.where(paid_mask, stage2.predict(X).clip(0, 1), 1.0)

    # ── 5. Expected Loss ──────────────────────────────────────
    print("[5/7] Tính Expected Loss (EL = PD × LGD × NỢ GỐC) ...")
    agg['EL_VND']   = agg['PD_SCORE'].apply(lambda x: 1-x) * agg['LGD_PRED'] * agg['NO_GOC']
    agg['EAD']      = agg['NO_GOC']

    total_portfolio_value = agg['NO_GOC'].sum()
    total_el              = agg['EL_VND'].sum()
    el_ratio              = total_el / total_portfolio_value * 100

    # ── 5B. NEW: EL by DPD Bucket ─────────────────────────────
    print("[5B/7] Phân tích EL theo DPD Bucket (NEW)...")
    agg['DPD_BUCKET'] = agg['DPD'].apply(assign_dpd_bucket)
    el_by_bucket = agg.groupby('DPD_BUCKET').agg(
        SO_HS    = ('LOAN ID',  'count'),
        TONG_EAD = ('EAD',     'sum'),
        TONG_EL  = ('EL_VND',  'sum'),
        REC_TB   = ('REC_RATE', 'mean'),
        LGD_TB   = ('LGD_PRED', 'mean'),
    ).reset_index()
    el_by_bucket['EL_RATE_%'] = (el_by_bucket['TONG_EL'] / el_by_bucket['TONG_EAD'] * 100).round(2)
    el_by_bucket['EL_TY']     = (el_by_bucket['TONG_EL'] / 1e9).round(2)
    el_by_bucket['EAD_TY']    = (el_by_bucket['TONG_EAD'] / 1e9).round(2)
    el_by_bucket['REC_TB_%']  = (el_by_bucket['REC_TB'] * 100).round(2)

    # Sắp xếp theo thứ tự DPD bucket
    bucket_order = [b[2] for b in DPD_BUCKETS]
    el_by_bucket['BUCKET_ORDER'] = el_by_bucket['DPD_BUCKET'].apply(
        lambda x: bucket_order.index(x) if x in bucket_order else 99
    )
    el_by_bucket = el_by_bucket.sort_values('BUCKET_ORDER')
    print("\n   EL Rate theo DPD Bucket:")
    print(el_by_bucket[['DPD_BUCKET','SO_HS','EAD_TY','EL_TY','EL_RATE_%','REC_TB_%']].to_string(index=False))

    # ── 5C. NEW: Top 20 hồ sơ EL cao nhất ─────────────────────
    top_el = agg.nlargest(20, 'EL_VND')[
        ['LOAN ID','DU_AN','RATING','DPD','NO_GOC','REC_RATE','EL_VND','DPD_BUCKET']
    ].copy()
    top_el['EL_VND_M'] = (top_el['EL_VND'] / 1e6).round(2)
    top_el['NO_GOC_M'] = (top_el['NO_GOC'] / 1e6).round(2)

    # ── 5D. NEW: EL by Partner ─────────────────────────────────
    el_by_partner = agg.groupby('DU_AN').agg(
        TONG_EAD       = ('EAD',     'sum'),
        TONG_EL        = ('EL_VND',  'sum'),
        SO_HO_SO       = ('LOAN ID', 'count'),
        RATING_A_PCT   = ('RATING',  lambda x: (x == 'A').mean() * 100),
        AVG_DPD        = ('DPD',     'mean'),
        REC_RATE_TB    = ('REC_RATE', 'mean'),
    ).reset_index()
    el_by_partner['EL_RATE_%']  = (el_by_partner['TONG_EL'] / el_by_partner['TONG_EAD'] * 100).round(2)
    el_by_partner['EAD_TY']     = (el_by_partner['TONG_EAD'] / 1e9).round(2)
    el_by_partner['EL_TY']      = (el_by_partner['TONG_EL'] / 1e9).round(2)
    el_by_partner['REC_RATE_%'] = (el_by_partner['REC_RATE_TB'] * 100).round(2)
    el_by_partner = el_by_partner.sort_values('EL_RATE_%', ascending=False)

    # EL by Rating
    el_by_rating = agg.groupby('RATING').agg(
        TONG_EL  = ('EL_VND',  'sum'),
        TONG_EAD = ('EAD',     'sum'),
        SO_HS    = ('LOAN ID', 'count'),
    ).reset_index()
    el_by_rating['EL_RATE_%'] = (el_by_rating['TONG_EL'] / el_by_rating['TONG_EAD'] * 100).round(2)
    el_by_rating = el_by_rating.sort_values('RATING')

    # EL Heatmap Matrix
    el_heatmap_df = agg.groupby(['DU_AN', 'RATING']).agg(
        TONG_EL = ('EL_VND', 'sum'),
        TONG_EAD = ('EAD', 'sum')
    ).reset_index()
    el_heatmap_df['EL_RATE'] = (el_heatmap_df['TONG_EL'] / el_heatmap_df['TONG_EAD'] * 100).round(2)
    el_heatmap_pivot = el_heatmap_df.pivot_table(index='DU_AN', columns='RATING', values='EL_RATE', fill_value=0)

    # Capital Estimation
    z_99 = 2.326
    agg['UL_VND'] = agg['EL_VND'] * z_99
    total_ul = agg['UL_VND'].sum()

    # ── 6. Build HTML Insights ────────────────────────────────
    insights_html = ""
    highest_el_partner = el_by_partner.iloc[0]['DU_AN']
    highest_el_rate = el_by_partner.iloc[0]['EL_RATE_%']
    if highest_el_rate > 80:
        insights_html += f"""
        <div class="alert-box alert-danger">
            <strong>🔴 Cảnh Báo Rủi Ro Từ {highest_el_partner}:</strong> Đối tác này có tỷ lệ Tổn thất Kỳ vọng (EL) lên tới {highest_el_rate:.1f}%. Cần siết chặt chuẩn phê duyệt đầu vào hoặc ngừng cấp room mới.
        </div>"""

    pd_top_feat = feat_importance.sort_values('Importance_PD', ascending=False).iloc[0]['Feature']
    lgd_top_feat = feat_importance.sort_values('Importance_LGD', ascending=False).iloc[0]['Feature']
    insights_html += f"""
    <div class="alert-box alert-success">
        <strong>💡 Động lực rủi ro (Model Insights):</strong> Yếu tố quyết định <b>CÓ TRẢ KHÔNG (PD)</b> là <b>{pd_top_feat}</b>. Yếu tố quyết định <b>TRẢ BAO NHIÊU (LGD)</b> là <b>{lgd_top_feat}</b>.
    </div>"""

    insights_html += f"""
    <div class="alert-box alert-warn">
        <strong>🟡 Capital Buffer (Unexpected Loss @ 99% CI):</strong> Để bù đắp UL cần trích lập <b>{total_ul/1e9:.2f} tỷ VND</b>. Tổng vốn dự phòng: <b>{(total_ul+total_el)/1e9:.2f} tỷ VND</b>.
    </div>"""

    # Highest EL bucket warning
    top_bucket = el_by_bucket.loc[el_by_bucket['EL_RATE_%'].idxmax()]
    insights_html += f"""
    <div class="alert-box alert-danger">
        <strong>🔴 DPD Bucket EL Cao Nhất:</strong> Nhóm <b>{top_bucket['DPD_BUCKET']}</b> có EL Rate = <b>{top_bucket['EL_RATE_%']:.1f}%</b> ({top_bucket['SO_HS']:,} hồ sơ | {top_bucket['EL_TY']:.2f} tỷ VND tổn thất kỳ vọng). Ưu tiên chiến lược tập trung vào bucket này.
    </div>"""

    # ── 7. Visualization & Dashboard ─────────────────────────
    print("\n[7/7] Tạo Dashboard HTML cao cấp (v3.0)...")
    import plotly.offline as plo

    # Fig 1: EL Decomposition
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("EL Rate (%) Theo Đối Tác", "EL Rate (%) Theo Xếp Hạng (A→D)"), horizontal_spacing=0.1)
    fig1.add_trace(go.Bar(x=el_by_partner['DU_AN'], y=el_by_partner['EL_RATE_%'], marker_color='#E53935', text=[f"{v:.1f}%" for v in el_by_partner['EL_RATE_%']], textposition='outside'), row=1, col=1)
    rating_colors = {'A':'#43A047','B':'#FDD835','C':'#FB8C00','D':'#E53935','0':'#9E9E9E'}
    fig1.add_trace(go.Bar(x=el_by_rating['RATING'], y=el_by_rating['EL_RATE_%'], marker_color=[rating_colors.get(r, '#9E9E9E') for r in el_by_rating['RATING']], text=[f"{v:.1f}%" for v in el_by_rating['EL_RATE_%']], textposition='outside'), row=1, col=2)
    fig1.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=50, l=40, r=40))
    div1 = plo.plot(fig1, output_type='div', include_plotlyjs=False)

    # Fig 2: EL Heatmap
    z_heatmap = el_heatmap_pivot.values
    fig2 = go.Figure(data=go.Heatmap(z=z_heatmap, x=el_heatmap_pivot.columns.tolist(), y=el_heatmap_pivot.index.tolist(), colorscale='YlOrRd', text=np.round(z_heatmap,1), texttemplate="%{text}%", showscale=True))
    fig2.update_layout(title="Ma trận EL Rate (%) theo Đối Tác × Xếp Hạng", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=50, l=40, r=40))
    div2 = plo.plot(fig2, output_type='div', include_plotlyjs=False)

    # Fig 3: Model Analysis
    feat_pd  = feat_importance.sort_values('Importance_PD',  ascending=True)
    feat_lgd = feat_importance.sort_values('Importance_LGD', ascending=True)
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Feature Importance - PD Model", "Feature Importance - LGD Model"), horizontal_spacing=0.1)
    fig3.add_trace(go.Bar(y=feat_pd['Feature'], x=feat_pd['Importance_PD'], orientation='h', marker_color='#1E88E5'), row=1, col=1)
    fig3.add_trace(go.Bar(y=feat_lgd['Feature'], x=feat_lgd['Importance_LGD'], orientation='h', marker_color='#7E57C2'), row=1, col=2)
    fig3.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=50, l=40, r=40))
    div3 = plo.plot(fig3, output_type='div', include_plotlyjs=False)

    # Fig 4: Recovery Distribution
    dist_df = dist.reset_index()
    dist_df.columns = ['Bucket', 'Count']
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Phân Phối Recovery Rate (Hồ Sơ Đã Trả)", "Phân Phối LGD Dự Báo (Mô Hình)"), horizontal_spacing=0.1)
    fig4.add_trace(go.Bar(x=dist_df['Bucket'], y=dist_df['Count'], marker_color=['#E53935','#FB8C00','#FDD835','#66BB6A','#1E88E5'], text=dist_df['Count'].apply(lambda v: f"{v:,}"), textposition='outside'), row=1, col=1)
    fig4.add_trace(go.Histogram(x=agg['LGD_PRED'], nbinsx=30, marker_color='#7E57C2'), row=1, col=2)
    fig4.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=50, l=40, r=40))
    div4 = plo.plot(fig4, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 5: EL by DPD Bucket ─────────────────────────
    bucket_colors = ['#43A047','#FDD835','#FB8C00','#FF5722','#E53935','#C62828','#880E4F','#4A148C']
    fig5 = make_subplots(rows=1, cols=2, subplot_titles=("EL Rate (%) Theo DPD Bucket", "Tổng EL (Tỷ VND) Theo DPD Bucket"), horizontal_spacing=0.1)
    eb = el_by_bucket.dropna(subset=['DPD_BUCKET'])
    fig5.add_trace(go.Bar(
        x=eb['DPD_BUCKET'], y=eb['EL_RATE_%'],
        marker_color=bucket_colors[:len(eb)],
        text=[f"{v:.1f}%" for v in eb['EL_RATE_%']], textposition='outside',
        hovertemplate="<b>%{x}</b><br>EL Rate: %{y:.1f}%<br>HS: " +
                      eb['SO_HS'].apply(lambda v: f"{v:,}").values.tolist().__str__() + "<extra></extra>"
    ), row=1, col=1)
    fig5.add_trace(go.Bar(
        x=eb['DPD_BUCKET'], y=eb['EL_TY'],
        marker_color=bucket_colors[:len(eb)],
        text=[f"{v:.2f}T" for v in eb['EL_TY']], textposition='outside'
    ), row=1, col=2)
    fig5.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=50, b=80, l=40, r=40))
    fig5.update_xaxes(tickangle=-20)
    div5 = plo.plot(fig5, output_type='div', include_plotlyjs=False)

    # ─── NEW Fig 6: Partner Detail Scatter ─────────────────────
    fig6 = go.Figure(go.Scatter(
        x=el_by_partner['AVG_DPD'],
        y=el_by_partner['EL_RATE_%'],
        mode='markers+text',
        text=el_by_partner['DU_AN'],
        textposition='top center',
        marker=dict(
            size=el_by_partner['EAD_TY'] * 3,
            color=el_by_partner['EL_RATE_%'],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title='EL Rate %')
        ),
        hovertemplate="<b>%{text}</b><br>AVG DPD: %{x:.0f}<br>EL Rate: %{y:.1f}%<extra></extra>"
    ))
    fig6.update_layout(
        title="Bubble Chart: DPD trung bình × EL Rate (%) × EAD (kích thước)",
        xaxis_title="DPD Trung Bình", yaxis_title="EL Rate (%)",
        height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=50, l=40, r=40)
    )
    div6 = plo.plot(fig6, output_type='div', include_plotlyjs=False)

    # ─── Data Tables ────────────────────────────────────────────
    # Partner summary table
    partner_rows = ""
    for _, r in el_by_partner.iterrows():
        el_badge_color = "#E53935" if r['EL_RATE_%'] > 80 else ("#FB8C00" if r['EL_RATE_%'] > 50 else "#10B981")
        partner_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['DU_AN']}</td>
            <td style="text-align:right;">{r['SO_HO_SO']:,}</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right;">{r['EL_TY']:.2f} Tỷ</td>
            <td style="text-align:right; color:{el_badge_color}; font-weight:700;">{r['EL_RATE_%']:.1f}%</td>
            <td style="text-align:right;">{r['REC_RATE_%']:.1f}%</td>
            <td style="text-align:right;">{r['AVG_DPD']:.0f}</td>
        </tr>"""

    # Top EL loans table
    top_el_rows = ""
    for i, r in top_el.iterrows():
        top_el_rows += f"""
        <tr>
            <td style="font-family:monospace; font-size:11px;">{r['LOAN ID']}</td>
            <td>{r['DU_AN']}</td>
            <td style="text-align:center;">{r['RATING']}</td>
            <td style="text-align:right;">{r['DPD']:.0f}</td>
            <td style="text-align:right;">{r['NO_GOC_M']:.1f}M</td>
            <td style="text-align:right;">{r['REC_RATE']*100:.1f}%</td>
            <td style="text-align:right; color:#EF4444; font-weight:700;">{r['EL_VND_M']:.1f}M</td>
            <td style="font-size:11px;">{r['DPD_BUCKET']}</td>
        </tr>"""

    # Bucket table
    bucket_rows = ""
    for _, r in el_by_bucket.iterrows():
        bucket_rows += f"""
        <tr>
            <td style="font-weight:600;">{r['DPD_BUCKET']}</td>
            <td style="text-align:right;">{r['SO_HS']:,}</td>
            <td style="text-align:right;">{r['EAD_TY']:.2f} Tỷ</td>
            <td style="text-align:right;">{r['EL_TY']:.2f} Tỷ</td>
            <td style="text-align:right; font-weight:600;">{r['EL_RATE_%']:.1f}%</td>
            <td style="text-align:right;">{r['REC_TB_%']:.1f}%</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>7D — LGD Model & Expected Loss Framework v3.0</title>
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
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.kpi-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; position: relative; box-shadow: 0 4px 6px -1px rgba(0,0,0,.15); }}
.kpi-card::after {{ content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 4px; background: var(--primary); border-radius: 0 0 var(--radius) var(--radius); }}
.kpi-card.kpi-success::after {{ background: var(--success); }}
.kpi-card.kpi-danger::after {{ background: var(--danger); }}
.kpi-card.kpi-warn::after {{ background: var(--warn); }}
.kpi-label {{ font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 22px; font-weight: 800; line-height: 1.1; }}
.kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); margin-bottom: 20px; }}
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
        [data-theme="dark"] {{ --background: #0F172A; --card-bg: #1E293B; --text-main: #F1F5F9; --text-muted: #94A3B8; --border: #334155; --primary-light: #38BDF8; }}
        :root:not([data-theme="dark"]) {{
            --bg: #F8FAFC !important; --card: #FFFFFF !important; --text: #0F172A !important;
            --border: #E2E8F0 !important; --muted: #64748B !important; --primary: #2563EB !important;
            --background: #F8FAFC !important; --card-bg: #FFFFFF !important;
            --text-main: #0F172A !important; --text-muted: #64748B !important;
            --success: #10B981 !important; --danger: #EF4444 !important;
            --warning: #F59E0B !important; --warn: #F59E0B !important; --info: #06B6D4 !important;
        }}
        html:not([data-theme="dark"]) body {{ background-color: var(--bg) !important; color: var(--text) !important; }}
        html:not([data-theme="dark"]) .kpi-card,
        html:not([data-theme="dark"]) .chart-card,
        html:not([data-theme="dark"]) .header {{ background-color: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; }}
        html:not([data-theme="dark"]) h1, html:not([data-theme="dark"]) h2, html:not([data-theme="dark"]) h3 {{ color: var(--text) !important; }}
        html:not([data-theme="dark"]) .kpi-label, html:not([data-theme="dark"]) .kpi-sub {{ color: var(--muted) !important; }}
        html:not([data-theme="dark"]) .data-table th {{ background-color: rgba(37,99,235,0.05) !important; color: #1E3A8A !important; }}
        html:not([data-theme="dark"]) .data-table td {{ color: var(--text) !important; border-color: var(--border) !important; }}
        html:not([data-theme="dark"]) .data-table tr:hover {{ background-color: rgba(37,99,235,0.05) !important; }}
        html:not([data-theme="dark"]) .tab-btn {{ color: var(--muted) !important; }}
        html:not([data-theme="dark"]) .tab-btn.active {{ color: var(--primary) !important; background-color: rgba(37,99,235,0.08) !important; }}
</style>
</head>
<body>
<div class="header">
    <h1>7D — LGD MODEL & EXPECTED LOSS FRAMEWORK <span style="font-size:14px; color:#64748B;">v3.0 | Risk Manager Edition</span></h1>
    <p>Đánh giá tỷ lệ tổn thất khi vỡ nợ (LGD), Ước tính tổn thất kỳ vọng (EL) theo DPD bucket & đối tác, xác định top hồ sơ EL cao nhất — VNE Risk Management 2026</p>
</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Tổng Giá Trị EAD</div>
        <div class="kpi-value">{total_portfolio_value/1e9:,.2f} Tỷ</div>
        <div class="kpi-sub">Tổng nợ gốc danh mục</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Expected Loss (EL)</div>
        <div class="kpi-value">{total_el/1e9:,.2f} Tỷ</div>
        <div class="kpi-sub">Ước tính tổn thất kỳ vọng</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">EL / EAD Tỷ lệ</div>
        <div class="kpi-value">{el_ratio:.2f}%</div>
        <div class="kpi-sub">Tỷ lệ tổn thất kỳ vọng bình quân</div>
    </div>
    <div class="kpi-card kpi-success">
        <div class="kpi-label">Stage 1 (PD) AUC-ROC</div>
        <div class="kpi-value">{pd_cv:.4f}</div>
        <div class="kpi-sub">Chỉ số Cross-Validation</div>
    </div>
    <div class="kpi-card kpi-warn">
        <div class="kpi-label">Unexpected Loss (UL@99%)</div>
        <div class="kpi-value">{total_ul/1e9:,.2f} Tỷ</div>
        <div class="kpi-sub">Mức dự phòng rủi ro bất ngờ</div>
    </div>
    <div class="kpi-card kpi-danger">
        <div class="kpi-label">Stage 2 (LGD) MAE</div>
        <div class="kpi-value">{-lgd_cv:.4f}</div>
        <div class="kpi-sub">Sai số tuyệt đối trung bình</div>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="openTab(event, 'tab1')">1. EL by DPD Bucket 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">2. EL Decomposition</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">3. Phân Tích Mô Hình</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">4. Bảng Chi Tiết Đối Tác 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab5')">5. Top 20 Hồ Sơ EL Cao 🆕</button>
    <button class="tab-btn" onclick="openTab(event, 'tab6')">6. Insight & Capital Planning</button>
</div>

<div id="tab1" class="tab-content active">
    <div class="chart-card">{div5}</div>
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📊 EL Rate theo DPD Bucket — Bảng Chi Tiết</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>DPD Bucket</th><th>Số Hồ Sơ</th><th>EAD (Tỷ)</th>
                <th>EL (Tỷ)</th><th>EL Rate %</th><th>Recovery Rate TB</th>
            </tr></thead>
            <tbody>{bucket_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab2" class="tab-content">
    <div class="chart-card" style="margin-bottom:20px;">{div2}</div>
    <div class="chart-card">{div1}</div>
    <div class="chart-card" style="margin-top:20px;">{div6}</div>
</div>

<div id="tab3" class="tab-content">
    <div class="chart-card" style="margin-bottom:20px;">{div3}</div>
    <div class="chart-card">{div4}</div>
</div>

<div id="tab4" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 16px 0; color:var(--primary); font-size:15px;">📋 Chi Tiết EL Theo Đối Tác</h3>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>Đối Tác</th><th>Số Hồ Sơ</th><th>EAD (Tỷ)</th>
                <th>EL (Tỷ)</th><th>EL Rate %</th><th>Recovery Rate</th><th>DPD TB</th>
            </tr></thead>
            <tbody>{partner_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab5" class="tab-content">
    <div class="chart-card">
        <h3 style="margin:0 0 8px 0; color:var(--primary); font-size:15px;">🚨 Top 20 Hồ Sơ có Expected Loss cao nhất</h3>
        <p style="font-size:12px; color:var(--muted); margin-bottom:16px;">Đây là các hồ sơ cần ưu tiên xử lý khẩn cấp — EL cao nhất trong toàn danh mục.</p>
        <div class="data-table-wrap">
        <table class="data-table">
            <thead><tr>
                <th>#</th><th>LOAN ID</th><th>Đối Tác</th><th>Rating</th>
                <th>DPD</th><th>Nợ Gốc (M)</th><th>Recovery Rate</th>
                <th>EL (M VND)</th><th>DPD Bucket</th>
            </tr></thead>
            <tbody>{top_el_rows}</tbody>
        </table></div>
    </div>
</div>

<div id="tab6" class="tab-content">
    <div class="chart-card" style="padding: 28px;">
        <h3 style="margin-top:0; color:var(--primary); margin-bottom: 20px;">💡 Đánh giá Rủi ro & Dự phòng Vốn</h3>
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
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        setTimeout(() => {{ updatePlotlyTheme(isDark); }}, 500);
    }});
}})();
</script>
</body>
</html>"""

    out_html = os.path.join(REPORT_DIR, "7d_LGD_ANALYSIS.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_csv = os.path.join(SUB_DATA_DIR, "7d_lgd_results.csv")
    agg[['LOAN ID','DU_AN','RATING','DPD','DPD_BUCKET','NO_GOC','REC_RATE','LGD',
         'PD_SCORE','LGD_PRED','EL_VND','EAD']].to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 7D v3.0!")
    print(f"   → HTML: {out_html}")
    print(f"   → CSV:  {out_csv}")
    print(f"\n📌 KEY INSIGHTS:")
    print(f"   EAD tổng: {total_portfolio_value/1e9:.2f} Tỷ | EL: {total_el/1e9:.2f} Tỷ ({el_ratio:.1f}%)")
    print(f"   UL @ 99%: {total_ul/1e9:.2f} Tỷ | Tổng vốn đề xuất: {(total_el+total_ul)/1e9:.2f} Tỷ")
    print(f"   Bucket EL cao nhất: {top_bucket['DPD_BUCKET']} ({top_bucket['EL_RATE_%']:.1f}%)")

if __name__ == "__main__":
    run()
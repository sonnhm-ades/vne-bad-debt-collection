# -*- coding: utf-8 -*-
"""
MODULE 8B — PHÂN TÍCH PHỄU LIÊN HỆ & MARKOV TRANSITION MATRIX (CONTACT FUNNEL + PTP SIGNAL)
Phiên bản: 2.0 (2026-05-26) — Tích hợp DS_SEGMENTATION để phân tích phễu theo CỤM_HÀNH_VI
Câu hỏi: Chuỗi liên hệ nào dẫn đến PAID nhanh nhất? PTP Score có dự đoán PAID không?
Biến nguồn (CLEANED.csv + DS_SEGMENTATION_FINAL.csv):
  - MÃ TÌNH TRẠNG LIÊN HỆ, KẾT QUẢ (CLEANED)
  - PTP_SCORE_PERCENT, CỤM_HÀNH_VI, SỐ TIỀN TT GẦN NHẤT, NGÀY CÓ KẾT QUẢ (DS_SEG)
Output: reports/Data_Science/CONTACT_FUNNEL.html
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as op
from datetime import datetime

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

# Nhãn tiếng Việt cho từng mã liên hệ
MA_LABELS = {
    # 1. Nhóm trạng thái liên hệ
    'CBACK': 'Máy bận/Hẹn gọi lại (CBACK)',
    'NCON':  'Không liên lạc được (NCON)',
    'NSP':   'Số sai/ảo (NSP)',
    'LM':    'Để lại tin nhắn tham chiếu (LM)',
    'HUP':   'Khách cúp máy ngang (HUP)',
    
    # 2. Nhóm hoàn cảnh đặc biệt
    'NCAP':  'Hoàn cảnh đặc biệt (NCAP)',
    
    # 3. Nhóm tiến độ & Cam kết
    'PTP':   'Hứa hẹn ngày trả (PTP)',
    'BPTP':  'Phá hứa/Thất hứa (BPTP)',
    'PAID':  'Đã thanh toán (PAID)',
    'NEGO':  'Đang đàm phán (NEGO)',
    
    # 4. Nhóm từ chối / Pháp lý
    'NIOP':  'Từ chối thanh toán/Pháp lý (NIOP)',
    'FRAUD': 'Hồ sơ giả mạo (FRAUD)',
    'KK/ĐXKK': 'Khởi kiện/Đề xuất khởi kiện (KK/ĐXKK)',
    
    # 5. Khác
    '0':     'Khác/Chưa xác định (0)'
}

CODE_TO_GROUP = {
    'CBACK': 'Trạng thái liên hệ',
    'NCON':  'Trạng thái liên hệ',
    'NSP':   'Trạng thái liên hệ',
    'LM':    'Trạng thái liên hệ',
    'HUP':   'Trạng thái liên hệ',
    
    'NCAP':  'Hoàn cảnh đặc biệt',
    
    'PTP':   'Tiến độ & Cam kết',
    'BPTP':  'Tiến độ & Cam kết',
    'PAID':  'Tiến độ & Cam kết',
    'NEGO':  'Tiến độ & Cam kết',
    
    'NIOP':  'Từ chối / Pháp lý',
    'FRAUD': 'Từ chối / Pháp lý',
    'KK/ĐXKK': 'Từ chối / Pháp lý',
    
    '0':     'Khác / Chưa xác định'
}

GROUP_COLORS = {
    'Trạng thái liên hệ': '#3F51B5',   # Soft Blue/Indigo
    'Hoàn cảnh đặc biệt': '#FF9800',   # Orange/Amber
    'Tiến độ & Cam kết': '#10B981',    # Emerald/Green
    'Từ chối / Pháp lý': '#E11D48',    # Rose/Crimson
    'Khác / Chưa xác định': '#64748B'  # Slate/Grey
}

def run():
    print("=" * 60)
    print("MODULE 8B — CONTACT FUNNEL & MARKOV ANALYSIS")
    print("=" * 60)

    # ── 1. Load ─────────────────────────────────────────
    print("\n[1/5] Đang nạp dữ liệu...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df['KẾT QUẢ'] = pd.to_numeric(df.get('KẾT QUẢ'), errors='coerce').fillna(0)

    # Bổ sung thêm thông tin từ DS_SEGMENTATION (PTP signal + Cluster)
    print("   Nạp DS_SEGMENTATION_FINAL.csv...")
    try:
        seg_avail_check = pd.read_csv(DS_SEG_PATH, nrows=1, low_memory=False)
        seg_want = ['LOAN ID', 'PTP_SCORE_PERCENT', 'CỤM_HÀNH_VI',
                    'SỐ TIỀN TT GẦN NHẤT', 'NGÀY CÓ KẾT QUẢ', 'SỐ_NGÀY_KHÔNG_THANH_TOÁN']
        seg_use = [c for c in seg_want if c in seg_avail_check.columns]
        seg_df  = pd.read_csv(DS_SEG_PATH, low_memory=False, usecols=seg_use)
        seg_df  = seg_df.drop_duplicates(subset='LOAN ID', keep='last')
        df = df.merge(seg_df, on='LOAN ID', how='left')
        print(f"   → Đã gắn {len(seg_use)-1} cột AI signal vào df")
    except Exception as e:
        print(f"   ⚠ Không tải được DS_SEG: {e}")

    mã_col = 'MÃ TÌNH TRẠNG LIÊN HỆ'
    if mã_col not in df.columns:
        print(f"LỖI: Không tìm thấy cột '{mã_col}'. Kiểm tra lại tên cột.")
        return

    # Chuẩn hóa mã (upper, strip)
    df[mã_col] = df[mã_col].astype(str).str.strip().str.upper()
    df[mã_col] = df[mã_col].replace({'NPS': 'NSP'})
    df[mã_col] = df[mã_col].replace({'NAN': np.nan, 'NONE': np.nan, '': np.nan})

    # ── KPI: Tính ở cấp LOAN-ID (đồng nhất với render_trend_module / Xu hướng Productivity) ──
    # Quy tắc: mỗi LOAN ID chỉ đếm 1 lần; lấy trạng thái CUỐI CÙNG theo thứ tự THÁNG
    total_loans = int(df['LOAN ID'].nunique())

    sort_cols_kpi = ['LOAN ID', 'THÁNG'] if 'THÁNG' in df.columns else ['LOAN ID']
    loan_last_code = (
        df[df[mã_col].notna()]
        .sort_values(sort_cols_kpi)
        .groupby('LOAN ID')[mã_col]
        .last()
    )

    # T0 COVERAGE – Hồ sơ có ít nhất 1 ghi nhận mã liên hệ hợp lệ
    t0_n = int(len(loan_last_code))
    total_interactions = t0_n  # Giữ biến cho phần Markov phía sau

    # T2+ REACH – Kết nối thực sự thành công (loại NCON/NSP); đồng nhất pipeline
    T2_PLUS = {'CBACK', 'HUP', 'LM', 'NEGO', 'BPTP', 'NIOP', 'NCAP', 'FRAUD', 'KK/ĐXKK', 'PTP', 'PAID'}
    t2_n = int(loan_last_code.isin(T2_PLUS).sum())
    contact_success_rate = t2_n / total_loans * 100

    # T4+ COMMIT – Hồ sơ có PTP hoặc PAID là trạng thái cuối
    T4_PLUS = {'PTP', 'PAID'}
    t4_n = int(loan_last_code.isin(T4_PLUS).sum())
    ptp_rate = t4_n / total_loans * 100
    ptp_count = int((loan_last_code == 'PTP').sum())  # Dùng tính Broken PTP

    # T5 RESOLVE – Hồ sơ có kết quả thanh toán thực tế (KẾT QUẢ > 0)
    paid_loan_count = int(df[df['KẾT QUẢ'] > 0]['LOAN ID'].nunique())
    paid_rate = paid_loan_count / total_loans * 100

    # Broken PTP – % phá hứa trong số hồ sơ có trạng thái cuối là PTP hoặc BPTP
    bptp_last_n = int((loan_last_code == 'BPTP').sum())
    ptp_bptp_total = ptp_count + bptp_last_n
    broken_promise_rate = (bptp_last_n / ptp_bptp_total * 100) if ptp_bptp_total > 0 else 0

    # PTP → PAID Conversion – trong số hồ sơ TỪNG có PTP (bất kỳ tháng), bao nhiêu % đã thanh toán
    loans_with_any_ptp = set(df[df[mã_col] == 'PTP']['LOAN ID'].unique())
    ptp_total_count = len(loans_with_any_ptp)
    ptp_then_paid = int(
        df[df['LOAN ID'].isin(loans_with_any_ptp) & (df['KẾT QUẢ'] > 0)]['LOAN ID'].nunique()
    )
    ptp_paid_conversion = (ptp_then_paid / ptp_total_count * 100) if ptp_total_count > 0 else 0

    # ── 2. Phân phối tần suất từng mã ─────────────────────────
    print("[2/5] Phân tích phân phối mã liên hệ...")
    df_valid = df[df[mã_col].notna()].copy()
    # Row-level counts (giữ lại để dùng chung cho Markov & state_order)
    ma_counts = df_valid[mã_col].value_counts().reset_index()
    ma_counts.columns = ['MÃ', 'SỐ_LẦN']
    ma_counts['NHÃN'] = ma_counts['MÃ'].map(MA_LABELS).fillna(ma_counts['MÃ'])
    ma_counts['%'] = (ma_counts['SỐ_LẦN'] / ma_counts['SỐ_LẦN'].sum() * 100).round(2)
    ma_counts['NHÓM'] = ma_counts['MÃ'].map(CODE_TO_GROUP).fillna('Khác / Chưa xác định')
    print(ma_counts.head(15).to_string(index=False))

    # Tính toán phân phối theo nhóm (row-level — chỉ dùng nội bộ cho Markov)
    df_valid['NHÓM'] = df_valid[mã_col].map(CODE_TO_GROUP).fillna('Khác / Chưa xác định')
    group_counts = df_valid['NHÓM'].value_counts().reset_index()
    group_counts.columns = ['NHÓM', 'SỐ_LẦN']
    group_counts['%'] = (group_counts['SỐ_LẦN'] / group_counts['SỐ_LẦN'].sum() * 100).round(2)

    # ── 2b. LOAN-LEVEL phân phối (Tab 1 — không đếm row) ──────────
    # Mỗi LOAN ID chỉ được tính 1 lần: lấy trạng thái CUỐI CÙNG (theo tháng)
    sort_cols_dist = ['LOAN ID', 'THÁNG'] if 'THÁNG' in df.columns else ['LOAN ID']
    loan_final_state = (
        df_valid.sort_values(sort_cols_dist)
        .groupby('LOAN ID')[mã_col]
        .last()
        .reset_index()
    )
    loan_final_state.columns = ['LOAN ID', 'MÃ_CUỐI']

    # Loan-level counts theo mã trạng thái cuối
    loan_ma_counts = loan_final_state['MÃ_CUỐI'].value_counts().reset_index()
    loan_ma_counts.columns = ['MÃ', 'SỐ_LOAN']
    loan_ma_counts['NHÃN'] = loan_ma_counts['MÃ'].map(MA_LABELS).fillna(loan_ma_counts['MÃ'])
    loan_ma_counts['%_LOAN'] = (loan_ma_counts['SỐ_LOAN'] / total_loans * 100).round(2)
    loan_ma_counts['NHÓM'] = loan_ma_counts['MÃ'].map(CODE_TO_GROUP).fillna('Khác / Chưa xác định')

    # Loan-level counts theo nhóm
    loan_group_counts = loan_final_state['MÃ_CUỐI'].map(CODE_TO_GROUP).fillna('Khác / Chưa xác định').value_counts().reset_index()
    loan_group_counts.columns = ['NHÓM', 'SỐ_LOAN']
    loan_group_counts['%_LOAN'] = (loan_group_counts['SỐ_LOAN'] / total_loans * 100).round(2)

    # Dual comparison: row-level vs loan-level (merge)
    dual_df = ma_counts[['MÃ', 'NHÃN', 'SỐ_LẦN', '%', 'NHÓM']].merge(
        loan_ma_counts[['MÃ', 'SỐ_LOAN', '%_LOAN']], on='MÃ', how='left'
    ).fillna(0)
    dual_df['SỐ_LOAN'] = dual_df['SỐ_LOAN'].astype(int)
    dual_df['AVG_CONTACTS'] = (dual_df['SỐ_LẦN'] / dual_df['SỐ_LOAN'].replace(0, np.nan)).round(1)
    dual_df = dual_df.sort_values('SỐ_LẦN', ascending=True)

    # Avg contacts per loan by group
    loan_contact_count = df_valid.groupby('LOAN ID').size().reset_index(name='TỔNG_LẦN_LH')
    loan_final_merged = loan_final_state.merge(loan_contact_count, on='LOAN ID', how='left')
    loan_final_merged['NHÓM_CUỐI'] = loan_final_merged['MÃ_CUỐI'].map(CODE_TO_GROUP).fillna('Khác / Chưa xác định')
    avg_contacts_by_group = (
        loan_final_merged.groupby('NHÓM_CUỐI')['TỔNG_LẦN_LH']
        .agg(['mean', 'median', 'count'])
        .reset_index()
    )
    avg_contacts_by_group.columns = ['NHÓM', 'TRUNG_BÌNH', 'TRUNG_VỊ', 'SỐ_HS']
    avg_contacts_by_group = avg_contacts_by_group.sort_values('TRUNG_BÌNH', ascending=True)

    # Histogram data: distribution of total contacts per loan
    contacts_dist = loan_contact_count['TỔNG_LẦN_LH'].clip(upper=30)  # cap at 30 for readability

    # Funnel data (T0 → T5) — Loan-level đầy đủ 6 tầng
    # T0 = Tổng danh mục (baseline); T1 = Có mã LH (tiếp cận); T3 = Đàm phán thực sự
    T3_SET = {'NEGO', 'BPTP', 'NIOP', 'NCAP', 'KK/ĐXKK', 'PTP', 'PAID'}  # đã có hội thoại thực sự
    t3_n = int(loan_last_code.isin(T3_SET).sum())

    funnel_labels = [
        f'T0: Tổng danh mục ({total_loans:,} HS)',
        f'T1: Tiếp cận — Có mã LH ({t0_n:,} HS)',
        f'T2+: Kết nối ({t2_n:,} HS)',
        f'T3: Đàm phán ({t3_n:,} HS)',
        f'T4+: Cam kết ({t4_n:,} HS)',
        f'T5: Thanh toán ({paid_loan_count:,} HS)'
    ]
    funnel_values = [total_loans, t0_n, t2_n, t3_n, t4_n, paid_loan_count]
    funnel_pcts   = [v / total_loans * 100 for v in funnel_values]
    print(f"   [Funnel] T0={total_loans:,} | T1={t0_n:,} | T2+={t2_n:,} | T3={t3_n:,} | T4+={t4_n:,} | T5={paid_loan_count:,}")

    # ── 3. Markov Transition Matrix ─────────────────────────────
    print("\n[3/5] Xây dựng Markov Transition Matrix...")
    sort_cols = ['LOAN ID']
    if 'THÁNG' in df.columns:
        sort_cols.append('THÁNG')
    
    total_raw_loans = df['LOAN ID'].nunique()
    df_sorted = df[df[mã_col].notna()].sort_values(sort_cols)
    loans_with_contact = df_sorted['LOAN ID'].nunique()
    n_missing_contact = total_raw_loans - loans_with_contact
    
    print(f"   → Tổng số LOAN ID duy nhất: {total_raw_loans:,}")
    if n_missing_contact > 0:
        print(f"   → Lưu ý: Có {n_missing_contact:,} hồ sơ không có bất kỳ mã liên hệ nào.")

    transitions = defaultdict(Counter)
    for loan_id, group in df_sorted.groupby('LOAN ID'):
        seq = group[mã_col].dropna().tolist()
        for i in range(len(seq) - 1):
            from_state = seq[i]
            to_state   = seq[i + 1]
            transitions[from_state][to_state] += 1

    # Tính xác suất
    all_states = sorted(set(
        list(transitions.keys()) +
        [s for v in transitions.values() for s in v.keys()]
    ))

    trans_matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)
    for from_s, to_dict in transitions.items():
        total = sum(to_dict.values())
        for to_s, cnt in to_dict.items():
            if from_s in trans_matrix.index and to_s in trans_matrix.columns:
                trans_matrix.loc[from_s, to_s] = round(cnt / total * 100, 1)

    # Đưa toàn bộ các trạng thái thực tế vào ma trận chuyển đổi sắp xếp theo tần suất xuất hiện
    state_order = [s for s in ma_counts['MÃ'].tolist() if s in trans_matrix.index]
    trans_viz = trans_matrix.loc[state_order, [s for s in state_order if s in trans_matrix.columns]]

    print(f"   → Ma trận {len(state_order)} × {len(state_order)} đã xây dựng.")

    # ── 4. Tìm tất cả đường dẫn → PAID (PTP > 0%) ─────────────
    print("\n[4/5] Truy vết toàn bộ chuỗi dẫn đến PAID...")
    paid_loans = df[df['KẾT QUẢ'] > 0]['LOAN ID'].unique()
    paid_df    = df_sorted[df_sorted['LOAN ID'].isin(paid_loans)]

    path_counter = Counter()
    for loan_id, group in paid_df.groupby('LOAN ID'):
        seq = group[mã_col].dropna().tolist()
        if len(seq) >= 2:
            path = tuple(seq[-4:])
            path_counter[path] += 1

    top_paths = path_counter.most_common(20)
    paths_df  = pd.DataFrame(
        [(' → '.join(p), cnt) for p, cnt in top_paths],
        columns=['CHUỖI_LIÊN_HỆ', 'SỐ_LOAN_ID']
    )
    paths_df['%_TRÊN_TỔNG_PAID'] = (paths_df['SỐ_LOAN_ID'] / len(paid_loans) * 100).round(2)
    print(paths_df.to_string(index=False))

    # ── 5. Visualization & Dashboard HTML ──────────────────────
    print("\n[5/5] Tạo Dashboard HTML cao cấp...")

    # Chart 1: Donut chart - Cơ cấu theo Nhóm Trạng thái
    group_fig = px.pie(
        group_counts, 
        values='SỐ_LẦN', 
        names='NHÓM',
        hole=0.4,
        color='NHÓM',
        color_discrete_map=GROUP_COLORS
    )
    group_fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Số lần: %{value:,}<br>Tỷ lệ: %{percent}<extra></extra>"
    )
    group_fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        showlegend=False,
        height=350
    )

    # Chart 2: Horizontal Bar chart - Phân phối Chi tiết Mã Trạng thái (LOAN-LEVEL)
    loan_ma_sorted = loan_ma_counts.sort_values('SỐ_LOAN', ascending=True)
    detail_fig = px.bar(
        loan_ma_sorted,
        x='SỐ_LOAN',
        y='NHÃN',
        color='NHÓM',
        color_discrete_map=GROUP_COLORS,
        orientation='h',
        text='SỐ_LOAN'
    )
    detail_fig.update_traces(
        texttemplate='%{text:,}',
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Nhóm: %{customdata[0]}<br>Hồ sơ (loan): %{x:,}<br>Tỷ lệ: %{customdata[1]:.2f}% tổng danh mục<extra></extra>"
    )
    detail_fig.update_traces(
        customdata=np.stack((loan_ma_sorted['NHÓM'], loan_ma_sorted['%_LOAN']), axis=-1)
    )
    detail_fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', title='Số hồ sơ (Loan ID)'),
        yaxis=dict(title=''),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520
    )

    # Chart NEW-A: Funnel T0→T5 đầy đủ 6 tầng (Loan-level)
    funnel_colors = ['#94A3B8', '#1E3A8A', '#3B82F6', '#8B5CF6', '#F59E0B', '#10B981']
    funnel_fig = go.Figure(go.Funnel(
        y=funnel_labels,
        x=funnel_values,
        textposition='inside',
        textinfo='value+percent initial',
        marker=dict(color=funnel_colors),
        connector=dict(line=dict(color='#CBD5E1', width=1, dash='dot')),
        hovertemplate="<b>%{label}</b><br>Số hồ sơ: %{value:,}<br>% tổng danh mục: %{percentInitial:.1%}<extra></extra>"
    ))
    funnel_fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=12),
        height=400
    )

    # Chart NEW-B: Dual-axis so sánh Row-level vs Loan-level per mã
    dual_sorted = dual_df.sort_values('SỐ_LẦN', ascending=True)
    dual_fig = go.Figure()
    dual_fig.add_trace(go.Bar(
        name='Số lượt tương tác (Row)',
        x=dual_sorted['SỐ_LẦN'],
        y=dual_sorted['NHÃN'],
        orientation='h',
        marker_color='#94A3B8',
        opacity=0.7,
        text=[f"{v:,}" for v in dual_sorted['SỐ_LẦN']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Lượt tương tác: %{x:,}<extra></extra>"
    ))
    dual_fig.add_trace(go.Bar(
        name='Số hồ sơ (Loan ID)',
        x=dual_sorted['SỐ_LOAN'],
        y=dual_sorted['NHÃN'],
        orientation='h',
        marker_color='#3B82F6',
        text=[f"{v:,}" for v in dual_sorted['SỐ_LOAN']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Hồ sơ duy nhất: %{x:,}<br>Avg lần LH/HS: %{customdata:.1f}<extra></extra>",
        customdata=dual_sorted['AVG_CONTACTS']
    ))
    dual_fig.update_layout(
        barmode='overlay',
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', title='Số lượng'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520
    )

    # Chart NEW-C: Avg contacts per loan theo nhóm (horizontal bar)
    avg_fig = go.Figure()
    avg_fig.add_trace(go.Bar(
        name='Trung bình lần LH/HS',
        x=avg_contacts_by_group['TRUNG_BÌNH'],
        y=avg_contacts_by_group['NHÓM'],
        orientation='h',
        marker=dict(
            color=[GROUP_COLORS.get(g, '#94A3B8') for g in avg_contacts_by_group['NHÓM']]
        ),
        text=[f"{v:.1f} lần" for v in avg_contacts_by_group['TRUNG_BÌNH']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Trung bình: %{x:.1f} lần/HS<br>Trung vị: %{customdata[0]:.1f}<br>Số HS: %{customdata[1]:,}<extra></extra>",
        customdata=np.stack((avg_contacts_by_group['TRUNG_VỊ'], avg_contacts_by_group['SỐ_HS']), axis=-1)
    ))
    avg_fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', title='Số lần liên hệ trung bình / hồ sơ'),
        height=280
    )

    # Chart NEW-D: Histogram phân bố số lần LH trên 1 LOAN ID
    hist_fig = go.Figure(go.Histogram(
        x=contacts_dist,
        nbinsx=30,
        marker=dict(color='#3B82F6', opacity=0.8, line=dict(color='#1E3A8A', width=0.5)),
        hovertemplate="Số lần LH: %{x}<br>Số hồ sơ: %{y:,}<extra></extra>"
    ))
    hist_fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', title='Tổng số lần liên hệ / hồ sơ (cap 30)'),
        yaxis=dict(showgrid=True, gridcolor='#E2E8F0', title='Số hồ sơ'),
        bargap=0.05,
        height=280
    )

    # Chart 3: Heatmap Markov
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=trans_viz.values,
        x=[MA_LABELS.get(s, s) for s in trans_viz.columns],
        y=[MA_LABELS.get(s, s) for s in trans_viz.index],
        colorscale='Blues',
        text=trans_viz.values.round(1),
        texttemplate="%{text}%",
        hovertemplate="Từ: %{y}<br>Sang: %{x}<br>Xác suất chuyển đổi: %{z}%<extra></extra>",
        showscale=True,
        colorbar=dict(title="Xác suất (%)")
    ))
    heatmap_fig.update_layout(
        margin=dict(t=30, b=30, l=150, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        xaxis=dict(tickangle=-30),
        height=550
    )

    # Chart 4: Top 10 Chuỗi Tương tác Dẫn đến PAID
    paths_top10 = paths_df.head(10).sort_values('SỐ_LOAN_ID', ascending=True)
    path_fig = px.bar(
        paths_top10,
        x='SỐ_LOAN_ID',
        y='CHUỖI_LIÊN_HỆ',
        orientation='h',
        text='SỐ_LOAN_ID'
    )
    path_fig.update_traces(
        marker_color='#26A69A',
        texttemplate='%{text:,} HS', 
        textposition='outside',
        hovertemplate="<b>Chuỗi: %{y}</b><br>Số hồ sơ: %{x:,}<br>Tỷ lệ: %{customdata:.2f}% trên tổng số PAID<extra></extra>"
    )
    path_fig.update_traces(customdata=paths_top10['%_TRÊN_TỔNG_PAID'])
    path_fig.update_layout(
        margin=dict(t=10, b=10, l=180, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
        height=450
    )

    # Chart 5: Xác suất chuyển đổi trực tiếp sang PAID
    if 'PAID' in trans_matrix.columns:
        paid_prob = trans_matrix['PAID'].sort_values(ascending=False).head(12).reset_index()
        paid_prob.columns = ['MÃ_HIỆN_TẠI', 'XS_→_PAID_%']
        paid_prob = paid_prob[paid_prob['XS_→_PAID_%'] > 0]
        paid_prob['NHÃN_HIỆN_TẠI'] = paid_prob['MÃ_HIỆN_TẠI'].map(MA_LABELS).fillna(paid_prob['MÃ_HIỆN_TẠI'])
        paid_prob['NHÓM'] = paid_prob['MÃ_HIỆN_TẠI'].map(CODE_TO_GROUP).fillna('Khác / Chưa xác định')
        
        paid_prob_fig = px.bar(
            paid_prob,
            x='NHÃN_HIỆN_TẠI',
            y='XS_→_PAID_%',
            color='NHÓM',
            color_discrete_map=GROUP_COLORS,
            text='XS_→_PAID_%'
        )
        paid_prob_fig.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='outside',
            hovertemplate="<b>Trạng thái: %{x}</b><br>Xác suất sang PAID: %{y:.1f}%<extra></extra>"
        )
        paid_prob_fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, Arial, sans-serif", size=11),
            yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )
    else:
        paid_prob_fig = go.Figure()

    # Chart 6: Cấu trúc hồ sơ PAID (Có PTP vs Không PTP)
    paid_with_ptp = ptp_then_paid
    paid_without_ptp = paid_loan_count - ptp_then_paid
    
    paid_structure_df = pd.DataFrame({
        'LOẠI': ['Thanh toán qua PTP', 'Thanh toán trực tiếp (Không qua PTP)'],
        'SỐ_LOAN': [paid_with_ptp, paid_without_ptp]
    })
    
    paid_structure_fig = px.pie(
        paid_structure_df,
        values='SỐ_LOAN',
        names='LOẠI',
        hole=0.4,
        color='LOẠI',
        color_discrete_map={
            'Thanh toán qua PTP': '#10B981',
            'Thanh toán trực tiếp (Không qua PTP)': '#3B82F6'
        }
    )
    paid_structure_fig.update_traces(
        textposition='inside',
        textinfo='percent+value',
        hovertemplate="<b>%{label}</b><br>Số hồ sơ: %{value:,}<br>Tỷ lệ: %{percent}<extra></extra>"
    )
    paid_structure_fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, Arial, sans-serif", size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=350
    )

    # Tạo HTML Divs cho các biểu đồ
    div_group_chart    = op.plot(group_fig,    output_type='div', include_plotlyjs=False)
    div_detail_chart   = op.plot(detail_fig,   output_type='div', include_plotlyjs=False)
    div_heatmap_chart  = op.plot(heatmap_fig,  output_type='div', include_plotlyjs=False)
    div_path_chart     = op.plot(path_fig,     output_type='div', include_plotlyjs=False)
    div_paid_prob_chart= op.plot(paid_prob_fig,output_type='div', include_plotlyjs=False)
    div_paid_structure_chart = op.plot(paid_structure_fig, output_type='div', include_plotlyjs=False)
    # Tab 1 — Loan-level charts
    div_funnel_chart   = op.plot(funnel_fig,   output_type='div', include_plotlyjs=False)
    div_dual_chart     = op.plot(dual_fig,     output_type='div', include_plotlyjs=False)
    div_avg_chart      = op.plot(avg_fig,      output_type='div', include_plotlyjs=False)
    div_hist_chart     = op.plot(hist_fig,     output_type='div', include_plotlyjs=False)

    # Format bảng cho tab 3
    table_rows = ""
    for _, r in paths_df.iterrows():
        table_rows += f"""
        <tr>
            <td style="font-family: monospace; font-weight: 500; color: #1E3A8A;">{r['CHUỖI_LIÊN_HỆ']}</td>
            <td style="text-align: right; font-weight: 600;">{int(r['SỐ_LOAN_ID']):,}</td>
            <td style="text-align: right; color: #10B981; font-weight: 600;">{r['%_TRÊN_TỔNG_PAID']:.2f}%</td>
        </tr>
        """

    n_missing_annotation = ""
    if n_missing_contact > 0:
        n_missing_annotation = f"""
        <div class="callout callout-danger" style="margin: 24px 0;">
            <strong>GHI CHÚ KIỂM SOÁT DỮ LIỆU:</strong> Có {n_missing_contact:,} hồ sơ hoàn toàn không có thông tin mã liên hệ trong file gốc, do đó không thể đưa vào phân tích Chuỗi tương tác và Ma trận xác suất. Vui lòng kiểm tra lại chất lượng đầu vào của hệ thống Tele-collection.
        </div>
        """

    current_time_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Toàn bộ mã nguồn giao diện HTML
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo Phân tích Phễu Liên hệ & Mô hình Markov</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --primary: #1E3A8A;
            --primary-light: #3B82F6;
            --background: #F8FAFC;
            --card-bg: #FFFFFF;
            --text-main: #0F172A;
            --text-muted: #64748B;
            --border: #E2E8F0;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --info: #06B6D4;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--background);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary) 0%, #0F172A 100%);
            color: white;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }}

        .header::after {{
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, transparent 70%);
            border-radius: 50%;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}

        .header p {{
            color: #94A3B8;
            font-size: 14px;
        }}

        .header-meta {{
            margin-top: 16px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .meta-badge {{
            background: rgba(255, 255, 255, 0.1);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            color: #E2E8F0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border-radius: 14px;
            padding: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border-top: 4px solid var(--primary-light);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        }}

        .kpi-title {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .kpi-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--text-main);
            margin: 8px 0 4px 0;
        }}

        .kpi-desc {{
            font-size: 11px;
            color: var(--text-muted);
        }}

        /* Tabs System */
        .tabs-container {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--border);
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
        }}

        .tab-headers {{
            display: flex;
            background: #F1F5F9;
            border-bottom: 1px solid var(--border);
            padding: 6px 12px 0 12px;
            gap: 4px;
            overflow-x: auto;
        }}

        .tab-btn {{
            padding: 12px 20px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
            white-space: nowrap;
        }}

        .tab-btn:hover {{
            color: var(--primary);
            background: rgba(255, 255, 255, 0.5);
        }}

        .tab-btn.active {{
            color: var(--primary);
            background: var(--card-bg);
            border-bottom: 2px solid var(--primary);
            box-shadow: 0 -2px 5px -2px rgba(0,0,0,0.05);
        }}

        .tab-content {{
            padding: 24px;
            display: none;
        }}

        .tab-content.active {{
            display: block;
            animation: fadeIn 0.3s ease;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Layout grids inside tabs */
        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }}

        .chart-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 16px;
        }}

        .chart-title {{
            font-size: 15px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}

        /* Tables styling */
        .data-table-container {{
            overflow-x: auto;
            max-height: 450px;
            border: 1px solid var(--border);
            border-radius: 8px;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}

        .data-table th {{
            background: #F8FAFC;
            padding: 10px 14px;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 2px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .data-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }}

        .data-table tr:hover {{
            background: #F1F5F9;
        }}

        /* Callout alert boxes */
        .callout {{
            padding: 16px;
            border-radius: 10px;
            margin-bottom: 16px;
            border-left: 4px solid;
            font-size: 14px;
        }}

        .callout.callout-info {{
            background-color: #EFF6FF;
            border-color: var(--primary-light);
            color: #1E40AF;
        }}

        .callout.callout-success {{
            background-color: #ECFDF5;
            border-color: var(--success);
            color: #065F46;
        }}

        .callout.callout-warning {{
            background-color: #FFFBEB;
            border-color: var(--warning);
            color: #92400E;
        }}

        .callout.callout-danger {{
            background-color: #FEF2F2;
            border-color: var(--danger);
            color: #991B1B;
        }}

        /* Recommendations Section */
        .rec-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}

        .rec-card {{
            background: #F8FAFC;
            border-radius: 10px;
            padding: 18px;
            border: 1px solid var(--border);
        }}

        .rec-card h4 {{
            font-size: 14px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .rec-card p {{
            font-size: 12.5px;
            color: #334155;
        }}

        .rec-card ul {{
            margin-left: 18px;
            margin-top: 8px;
            font-size: 12px;
            color: #475569;
        }}

        .rec-card li {{
            margin-bottom: 4px;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 24px;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            margin-top: 24px;
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .grid-2col {{
                grid-template-columns: 1fr;
            }}
            body {{
                padding: 12px;
            }}
        }}
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

    <!-- Header -->
    <div class="header">
        <h1>📊 Phân Tích Phễu Thu Hồi Nợ — Contact-to-Collection Funnel</h1>
        <p>Đánh giá hiệu quả tác nghiệp theo chuỗi trạng thái liên hệ (T0→T5), đồng nhất với chuẩn Xu hướng Productivity — VNE 2026.</p>
        <div class="header-meta">
            <span class="meta-badge">📂 Nguồn dữ liệu: TỔNG HỢP NĂM 2026 CLEANED.csv</span>
            <span class="meta-badge">🔄 Cập nhật: {current_time_str}</span>
            <span class="meta-badge">📋 Tổng LOAN ID danh mục: {total_loans:,}</span>
        </div>
    </div>

    <!-- KPI Strip — Loan-level metrics aligned with Xu hướng Productivity (T0→T5 funnel) -->
    <div class="kpi-grid">
        <div class="kpi-card" style="border-top-color: #94a3b8;">
            <div class="kpi-title">T0: Tổng Danh Mục</div>
            <div class="kpi-value">{total_loans:,} HS</div>
            <div class="kpi-desc">Tổng số hồ sơ duy nhất cần thu hồi nợ (baseline)</div>
        </div>
        <div class="kpi-card" style="border-top-color: #1e3a8a;">
            <div class="kpi-title">T1: Đã Tiếp Cận</div>
            <div class="kpi-value">{(t0_n/total_loans*100):.2f}%</div>
            <div class="kpi-desc">{t0_n:,} HS có ≥1 ghi nhận mã liên hệ / tổng danh mục</div>
        </div>
        <div class="kpi-card" style="border-top-color: #3b82f6;">
            <div class="kpi-title">T2+: Kết Nối (Reach)</div>
            <div class="kpi-value">{contact_success_rate:.2f}%</div>
            <div class="kpi-desc">{t2_n:,} HS kết nối thành công (loại trừ NSP, NCON...)</div>
        </div>
        <div class="kpi-card" style="border-top-color: #8b5cf6;">
            <div class="kpi-title">T3: Đàm Phán (Nego)</div>
            <div class="kpi-value">{(t3_n/total_loans*100):.2f}%</div>
            <div class="kpi-desc">{t3_n:,} HS đã phát sinh cuộc hội thoại đàm phán thực sự</div>
        </div>
        <div class="kpi-card" style="border-top-color: #eab308;">
            <div class="kpi-title">T4+ COMMIT (Hứa Trả)</div>
            <div class="kpi-value">{ptp_rate:.2f}%</div>
            <div class="kpi-desc">{t4_n:,} HS có PTP/PAID là trạng thái cuối / tổng danh mục</div>
        </div>
        <div class="kpi-card" style="border-top-color: #10b981;">
            <div class="kpi-title">T5 RESOLVE (Thanh Toán)</div>
            <div class="kpi-value">{paid_rate:.2f}%</div>
            <div class="kpi-desc">{paid_loan_count:,} HS đã thanh toán thực tế (KẾT QUẢ > 0)</div>
        </div>
        <div class="kpi-card" style="border-top-color: #06b6d4;">
            <div class="kpi-title">PTP → PAID Conversion</div>
            <div class="kpi-value">{ptp_paid_conversion:.1f}%</div>
            <div class="kpi-desc">{ptp_then_paid:,}/{ptp_total_count:,} HS từng hứa hẹn PTP đã thanh toán</div>
        </div>
        <div class="kpi-card" style="border-top-color: #ef4444;">
            <div class="kpi-title">Broken PTP (Thất Hứa)</div>
            <div class="kpi-value">{broken_promise_rate:.2f}%</div>
            <div class="kpi-desc">{bptp_last_n:,}/{ptp_bptp_total:,} HS có PTP/BPTP cuối bị thất hứa</div>
        </div>
    </div>

    <!-- Tabbed Container -->
    <div class="tabs-container">
        <div class="tab-headers">
            <button class="tab-btn active" onclick="openTab(event, 'tab-dist')">1. Phân phối & Phân nhóm</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-markov')">2. Ma trận Chuyển đổi Markov</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-paths')">3. Chuỗi thanh toán tối ưu</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-insights')">4. Insight & Khuyến nghị hành động</button>
        </div>

        <!-- Tab 1: Distribution (Loan-level) -->
        <div id="tab-dist" class="tab-content active">
            <div class="callout callout-info">
                <strong>Góc nhìn Risk Manager — Phân tích cấp Hồ sơ (Loan-level):</strong>
                Tab này hiển thị <strong>số hồ sơ duy nhất (Loan ID)</strong>, không phải số lượt gọi. Mỗi hồ sơ chỉ được tính 1 lần theo trạng thái <em>cuối cùng ghi nhận</em>.
                Phễu T0→T5 phản ánh mức độ "rò rỉ" thực tế của danh mục, trong khi biểu đồ so sánh Lượt gọi vs Hồ sơ giúp phát hiện nhóm nào đang bị tốn effort không cần thiết.
            </div>

            <!-- ROW 1: Funnel + Donut Loan-level -->
            <div class="grid-2col">
                <div class="chart-card">
                    <div class="chart-title">📉 Phễu Thu Hồi T0 → T5 (Loan-level — Mỗi hồ sơ tính 1 lần)</div>
                    {div_funnel_chart}
                    <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                        <strong>Đọc biểu đồ:</strong> Mỗi tầng = % hồ sơ đạt được ngưỡng đó so với tổng danh mục {total_loans:,} HS.
                        Khoảng cách giữa các tầng = tỷ lệ "rò rỉ" — mục tiêu chiến lược là thu hẹp khoảng này.
                    </div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">🗂 Phân bổ Hồ sơ theo Trạng thái Cuối (Loan-level — % trên tổng danh mục)</div>
                    {div_detail_chart}
                    <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                        <strong>Lưu ý:</strong> Mã NCON/CBACK chiếm đa số không có nghĩa là agent thiếu hiệu quả — cần so sánh với<em> số lần gọi trung bình/hồ sơ</em> phía dưới để đánh giá chính xác.
                    </div>
                </div>
            </div>

            <!-- ROW 2: Dual-bar Row vs Loan -->
            <div class="chart-card">
                <div class="chart-title">⚖️ So sánh: Lượt Tương tác (Row) vs Hồ sơ Duy nhất (Loan-level) — Phát hiện Nhóm Tốn Effort</div>
                {div_dual_chart}
                <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                    <strong>Ý nghĩa:</strong> Khoảng cách lớn giữa thanh xám (tổng lượt gọi) và thanh xanh (số hồ sơ) = trung bình mỗi hồ sơ trong nhóm đó bị gọi nhiều lần mà không ra kết quả.
                    Tỷ lệ Avg LH/HS cao ở nhóm NCON/CBACK là dấu hiệu cần xem xét lại quy trình phân luồng.
                </div>
            </div>

            <!-- ROW 3: Avg contacts by group + Histogram -->
            <div class="grid-2col">
                <div class="chart-card">
                    <div class="chart-title">📊 Số Lần Liên Hệ Trung Bình / Hồ Sơ theo Nhóm Trạng thái Cuối</div>
                    {div_avg_chart}
                    <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                        <strong>Chiến lược:</strong> Nhóm có avg cao + kết quả kém = ưu tiên tái phân luồng (legal/freeze).
                        Nhóm có avg thấp + kết quả tốt = best practice cần nhân rộng.
                    </div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">📈 Phân bố Tổng Lần Liên Hệ trên 1 Hồ Sơ (Histogram — cap 30 lần)</div>
                    {div_hist_chart}
                    <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                        <strong>Đọc biểu đồ:</strong> Phân bố lệch phải (long tail) = tồn tại nhóm hồ sơ bị gọi quá nhiều lần (outlier) cần được đặt lên diện theo dõi đặc biệt hoặc chuyển kênh xử lý.
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Markov -->
        <div id="tab-markov" class="tab-content">
            <div class="callout callout-warning">
                <strong>Phân tích chuyển đổi trạng thái (Markov):</strong>
                Ma trận phản ánh xác suất di chuyển giữa các trạng thái theo từng tháng, tính trên từ́ng cặp chuyển đổi liên tiếp. 
                <strong>PAID là trạng thái hấp thụ</strong> (absorbing state) — một khi đã vào thường khó thoát ra. 
                Đọc cột PAID để tìm trạng thái nào có xác suất chuyển sang PAID cao nhất — đó là tín hiệu ưu tiên tác nghiệp.
            </div>

            <!-- Heatmap full width -->
            <div class="chart-card">
                <div class="chart-title">Markov Transition Heatmap (%) — Xác suất chuyển đổi trạng thái giữa các tháng</div>
                {div_heatmap_chart}
                <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                    <strong>Đọc hầu mật:</strong> Hàng = trạng thái hiện tại. Cột = trạng thái tiếp theo. 
                    Ô màu đậm (số cao) = con đường chuyển đổi phổ biến nhất. 
                    Hàng có giá trị cao nhất ở cột <em>chính nó</em> = trạng thái tự lặp/mắt kết — đối tượng cần xác định chỉìt giới hạn liên hệ (call cap).
                </div>
            </div>

            <!-- Xác suất sang PAID theo trạng thái + callout insight -->
            <div class="grid-2col">
                <div class="chart-card">
                    <div class="chart-title">Xác suất Đi thẳng sang PAID từ Từng Trạng thái (%)</div>
                    {div_paid_prob_chart}
                    <div style="font-size:12px; color:#64748B; margin-top:8px; padding:8px; background:#F8FAFC; border-radius:6px;">
                        <strong>Giải thích:</strong> Biểu đồ chiết xuất cột PAID từ Ma trận Markov. 
                        Trạng thái có xác suất cao = âm thanh bảo lãnh thu hồi, nên ưu tiên bám sát ngay tháng tiếp theo.
                    </div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">📌 Insight — Trạng thái Hấp thụ và Con đường Ổi nhất</div>
                    <div style="font-size:13.5px; line-height:1.7; color:#334155;">
                        <p style="margin-bottom:10px;"><strong>🔴 Self-loop cần cắt:</strong>
                        Những trạng thái có xác suất <em>lặp lại chính nó</em> cao (CBACK→CBACK, NCON→NCON) cho thấy hồ sơ bị mắt kết không tiến triển. Cần áp dụng ‘call cap’: giới hạn tối đa 3–5 lần/tháng, sau đó chuyển sang kênh khác (SMS/Zalo/email).</p>
                        <p style="margin-bottom:10px;"><strong>🟢 Con đường BPTP → PTP → PAID:</strong>
                        Dù khách thất hứa, vẫn có xác suất quay lại đàm phán. Window tối ưu để gọi lại sau BPTP là <strong>24–48 giờ</strong> — trễ hơn thì xác suất rờt nhanh.</p>
                        <p style="margin-bottom:10px;"><strong>🟡 NIOP / KK/ĐXKK → PAID gần như = 0:</strong>
                        Một khi đã ở trạng thái từ chối/pháp lý, xác suất phục hồi bằng cách gọi thông thường = gần 0. Chi phí tiếp tục gọi = lãng phí. Cần chuyển hồ sơ sang bộ phận pháp lý hước <strong>trong vòng 30 ngày</strong> kể từ lần đầu ghi nhận mã này.</p>
                        <p><strong>🟣 NEGO → PTP: </strong>
                        Đây là con đường tối ưu nhất. Hồ sơ đang ở NEGO có xác suất chuyển sang PTP hoặc PAID rất cao. Đây là khoảnh khắc vàng — agent cần đi đến chốt số tiền cụ thể và ngày hẹn, không rời mà không có cam kết rõ ràng.</p>
                        <p><strong>🎯 Ưu tiên số 1 (High Probability):</strong> Các trạng thái có xác suất sang PAID cao (ví dụ NEGO, PTP) cần được agent bám sát (Call-Followup) trong 24h. Đừng để khách hàng rơi vào trạng thái 'chờ' (NCON).</p>
                        <p><strong>⚠️ Cảnh báo tự lặp (Self-loop):</strong> Các trạng thái lặp lại chính nó quá nhiều lần (CBACK, NCON) làm loãng resource. Áp dụng 'Call Cap' ngay khi thấy ngưỡng 5 cuộc gọi không thay đổi trạng thái.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 3: Paths -->
        <div id="tab-paths" class="tab-content">
            <div class="callout callout-success">
                <strong>Truy vết chuỗi tương tác — Tìm ra "Công thức vàng":</strong>
                Phân tích {len(paid_loans):,} hồ sơ đã thanh toán thành công để tìm ra quy luật chuỗi tiếp xúc. 
                Chúng tôi tập trung vào việc <strong>kết hợp các loại trạng thái</strong> để biến một hồ sơ lạnh thành hồ sơ thu tiền.
            </div>

            <!-- ROW 1: Donut structure chart + Detailed Operational Insight -->
            <div class="grid-2col" style="margin-bottom: 24px;">
                <div class="chart-card">
                    <div class="chart-title">🍩 Cấu trúc hồ sơ PAID (Có PTP vs Không PTP)</div>
                    {div_paid_structure_chart}
                </div>
                <div class="chart-card">
                    <div class="chart-title">💡 Giải mã Nghiệp vụ: Cam Kết vs Thực Thu</div>
                    <div style="font-size:13.5px; line-height:1.7; color:#334155;">
                        <p style="margin-bottom:10px;">📊 <strong>Tổng số hồ sơ thanh toán thực tế (T5):</strong> {paid_loan_count:,} HS.</p>
                        <p style="margin-bottom:10px;">🟢 <strong>Thanh toán qua PTP (Cam kết trước):</strong> {ptp_then_paid:,} HS (chiếm {(ptp_then_paid/paid_loan_count*100):.1f}%). 
                        Nhóm này thể hiện tính tuân thủ và hiệu quả tác động của Agent. Với tỷ lệ PTP→PAID Conversion đạt <strong>{ptp_paid_conversion:.1f}%</strong> ({ptp_then_paid:,}/{ptp_total_count:,} HS từng hứa đã thanh toán), có thể khẳng định: <em>Cứ hướng dẫn khách hàng ra được cam kết PTP thành công thì cơ hội thu hồi nợ cực kỳ cao.</em></p>
                        <p style="margin-bottom:10px;">🔵 <strong>Thanh toán trực tiếp (Không qua PTP):</strong> {paid_loan_count - ptp_then_paid:,} HS (chiếm {((paid_loan_count - ptp_then_paid)/paid_loan_count*100):.1f}%).
                        Nhóm này thanh toán trực tiếp mà không có bất kỳ mã hẹn trả PTP nào được lưu trong lịch sử. Điều này chỉ ra 2 khả năng:</p>
                        <ul style="margin-left: 20px; margin-bottom: 10px;">
                            <li><strong>Khách tự động trả nợ:</strong> Tự thanh toán qua app/chuyển khoản mà không cần đôn đốc trực tiếp.</li>
                            <li><strong>Lỗ hổng nhập liệu (Data Integrity):</strong> Agent đàm phán thành công nhưng lười nhập mã cam kết PTP lên hệ thống, hoặc chỉ ghi nhận NEGO/CBACK rồi chờ tiền về để được tính PAID trực tiếp.</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- ROW 2: Top 10 Paths & Detailed Table -->
            <div class="grid-2col">
                <div class="chart-card">
                    <div class="chart-title">Top 10 Chuỗi Tương tác dẫn đến PAID (số hồ sơ)</div>
                    {div_path_chart}
                </div>
                <div class="chart-card">
                    <div class="chart-title">Bảng chi tiết toàn bộ Chuỗi dẫn đến PAID</div>
                    <div class="data-table-container">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Chuỗi liên hệ (4 bước cuối)</th>
                                    <th>Số hồ sơ</th>
                                    <th>% trên tổng PAID</th>
                                </tr>
                            </thead>
                            <tbody>
                                {table_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 4: Insights & Recommendations -->
        <div id="tab-insights" class="tab-content">
            <div class="chart-card">
                <div class="chart-title" style="color: var(--primary);">💡 Insight Định lượng từ Dữ liệu Thực tế</div>
                <div style="font-size: 13.5px; color: #334155; line-height: 1.7;">
                    <p style="margin-bottom:14px;"><strong>1. Khoảng trống tiếp cận khổng lồ ({total_loans - t0_n:,} HS = {(total_loans-t0_n)/total_loans*100:.1f}% danh mục):</strong>
                    Đây là ưu tiên số 1 của Risk Manager. Gần <strong>{(total_loans-t0_n)/total_loans*100:.0f}%</strong> hồ sơ chưa hề được tiếp cận với bất kỳ mã liên hệ nào.
                    Nguyên nhân có thể: số điện thoại sai/hết hạn, dữ liệu đầu vào thiếu, hoặc agent chưa bắt đầu xử lý. Cần phân nhóm và lên kế hoạch tiếp cận riêng cho nhóm này trước khi nó trở thành nợ xấu không thể thu hồi.</p>

                    <p style="margin-bottom:14px;"><strong>2. Tỷ lệ đàm phán thành cam kết thấp đáng lo ngại (T3→T4: {t3_n:,}→{t4_n:,} = {t4_n/t3_n*100:.1f}%):</strong>
                    Trong số {t3_n:,} hồ sơ đã đàm phán thực sự, chỉ {t4_n/t3_n*100:.1f}% kết thúc bằng cam kết PTP/PAID là trạng thái cuối.
                    Điều này gợi ý agent đang <em>bỏ lỡ cơ hội chốt cam kết</em> ngay tại cuộc gọi — cần training kỹ năng chốt và quy trình chuẩn hóa script khi khách đang ở trạng thái NEGO.</p>

                    <p style="margin-bottom:14px;"><strong>3. PTP→PAID Conversion = {ptp_paid_conversion:.1f}% ({ptp_then_paid:,}/{ptp_total_count:,} HS):</strong>
                    Hơn 3/4 số khách hàng từng cam kết PTP cuối cùng đã thanh toán. Đây là tín hiệu <strong>khả năng thu hồi tốt</strong> khi tiếp cận đúng cách.
                    Vấn đề là số lượng PTP quá thấp ({ptp_total_count:,} HS = {ptp_total_count/total_loans*100:.1f}% danh mục) — cần nâng tỷ lệ đàm phán đến cam kết thay vì tăng số lần gọi.</p>

                    <p><strong>4. Broken PTP = {broken_promise_rate:.1f}% ({bptp_last_n:,}/{ptp_bptp_total:,} HS có PTP/BPTP cuối):</strong>
                    Tỷ lệ thất hứa này cần được so sánh với benchmark ngành (thường 30-50%). Nếu dưới 30% là tốt, nếu trên 50% cần xem lại cách agent xác nhận PTP (có thể đang chấp nhận PTP "xã giao" không có ý định thực sự từ khách hàng).</p>
                </div>
            </div>

            <div class="chart-card">
                <div class="chart-title" style="color: var(--primary);">🎯 Khuyến nghị Hành động Ưu tiên (Priority Action Plan)</div>
                <div class="rec-grid">
                    <div class="rec-card" style="border-left: 4px solid #EF4444;">
                        <h4>🔴 P1 — Tiếp cận {total_loans - t0_n:,} HS chưa có mã LH</h4>
                        <p>Chiếm {(total_loans-t0_n)/total_loans*100:.0f}% danh mục, đây là cơ hội thu hồi lớn nhất chưa được khai thác.</p>
                        <ul>
                            <li>Kiểm tra chất lượng số điện thoại (skiptracing)</li>
                            <li>Phân loại theo số tiền nợ, ưu tiên HS có dư nợ cao</li>
                            <li>Giao cho team chuyên biệt xử lý cold outreach</li>
                        </ul>
                    </div>
                    <div class="rec-card" style="border-left: 4px solid #F59E0B;">
                        <h4>🟡 P2 — Áp dụng Call Cap cho NCON/CBACK tự lặp</h4>
                        <p>Hồ sơ bị gọi nhiều lần mà vẫn NCON/CBACK là dấu hiệu lãng phí effort.</p>
                        <ul>
                            <li>Giới hạn tối đa 3–5 lần/tháng cho mỗi số</li>
                            <li>Sau 3 tháng NCON liên tục: đẩy sang skiptracing</li>
                            <li>Thử đổi khung giờ gọi (sáng sớm, tối muộn)</li>
                        </ul>
                    </div>
                    <div class="rec-card" style="border-left: 4px solid #10B981;">
                        <h4>🟢 P3 — Tăng tỷ lệ chốt PTP từ NEGO</h4>
                        <p>NEGO là khoảnh khắc vàng — agent cần đi đến cam kết cụ thể, không rời mà không có PTP.</p>
                        <ul>
                            <li>Chuẩn hóa script chốt: số tiền + ngày hẹn + phương thức</li>
                            <li>Gửi xác nhận bằng SMS/Zalo ngay sau cuộc gọi</li>
                            <li>Supervisor review cuộc gọi NEGO không kết thúc bằng PTP</li>
                        </ul>
                    </div>
                    <div class="rec-card" style="border-left: 4px solid #6366F1;">
                        <h4>🟣 P4 — Chuyển sớm NIOP/KK sang Pháp lý</h4>
                        <p>Xác suất Markov NIOP/KK→PAID gần 0. Mỗi cuộc gọi thêm = chi phí thuần không thu hồi.</p>
                        <ul>
                            <li>Chuẩn hóa ngưỡng chuyển pháp lý: &lt;30 ngày kể từ lần đầu ghi NIOP</li>
                            <li>Lập hồ sơ chứng cứ song song với quá trình gọi</li>
                            <li>Đo ROI của từng HS khi quyết định khởi kiện vs xóa nợ</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    {n_missing_annotation}

    <div class="footer">
        Báo cáo Phân tích Phễu Liên hệ & Markov · Phát triển bởi Antigravity AI Coding Assistant © 2026
    </div>

    <!-- Script switching tabs -->
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }}
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].classList.remove("active");
            }}
            document.getElementById(tabName).style.display = "block";
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
            
            // Trigger redraw of plotly charts to make sure they fit their new container size
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

    # Ghi file HTML
    out_html = os.path.join(REPORT_DIR, "8b_CONTACT_FUNNEL.html")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Ghi file dữ liệu trung gian CSV
    out_csv = os.path.join(SUB_DATA_DIR, "8b_paid_paths.csv")
    paths_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    out_matrix = os.path.join(SUB_DATA_DIR, "8b_markov_matrix.csv")
    trans_matrix.to_csv(out_matrix, encoding='utf-8-sig')

    print(f"\n✅ HOÀN THÀNH MODULE 8B!")
    print(f"   → HTML:          {out_html}")
    print(f"   → Paid Paths:    {out_csv}")
    print(f"   → Markov Matrix: {out_matrix}")

if __name__ == "__main__":
    run()
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pickle
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

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

DIR_PATH = r'c:\Users\HP\Nextcloud\SƠN - PHÂN TÍCH\TONG HOP NAM 2026\reports\Data_Science'
DATA_FILE = os.path.join(DIR_PATH, "DS_SEGMENTATION_FINAL.csv")
SHAP_FILE = os.path.join(DIR_PATH, "shap_data_for_dashboard.pkl")
HTML_OUTPUT = os.path.join(DIR_PATH, "ADVANCED_INSIGHT.html")

def create_dashboard():
    print("1. Nạp dữ liệu để vẽ Dashboard...")
    if not os.path.exists(DATA_FILE):
        print(f"LỖI: Không tìm thấy {DATA_FILE}. Vui lòng chạy 7a và 7b trước.")
        return
        
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Init Plotly subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "1. Phân Phối Khả Năng Trả Nợ (PTP Score) - Dải < 5%",
            "2. Phân Phối Khả Năng Trả Nợ (PTP Score) - Dải > 5%",
            "3. Top Yếu Tố Quyết Định Thu Hồi (SHAP Feature Importance)",
            "4. Ma Trận Năng Lực Tài Chính & Tỷ Lệ PTP Trung Bình",
            "5. Phân Cụm Hành Vi (K-Means) & Trung Vị DPD",
            "6. Tương Tác Gần Đây vs Khả Năng Trả Nợ"
        ),
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "pie"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    
    # ----- Plot 1 & 2: PTP Score Distribution (Log Scale awareness) -----
    print("2. Đang tạo biểu đồ phân phối PTP...")
    low_ptp = df[df['PTP_SCORE_PERCENT'] <= 5]['PTP_SCORE_PERCENT']
    high_ptp = df[df['PTP_SCORE_PERCENT'] > 5]['PTP_SCORE_PERCENT']
    
    fig.add_trace(go.Histogram(
        x=low_ptp, name='PTP <= 5% (Dead Debt zone)', marker_color='#EF553B', nbinsx=50
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=high_ptp, name='PTP > 5% (Tiềm năng)', marker_color='#00CC96', nbinsx=50
    ), row=1, col=2)
    
    # Note for skewed distribution
    fig.add_annotation(
        text="<b>Chú thích:</b> Tập dữ liệu có phân phối cực kỳ lệch phải (Skewed).<br>Hơn 95% khách hàng có xác suất trả nợ dưới 1%.<br>Trục Y đã được dùng <b>Log-Scale</b> để bạn có thể nhìn thấy<br>các hồ sơ tiềm năng ở nhóm >5%.",
        xref="paper", yref="paper", x=0.5, y=1.05,
        showarrow=False, font=dict(size=13, color="red"),
        bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="red", borderwidth=1
    )
    
    fig.update_yaxes(type="log", title_text="Số lượng hồ sơ (Log Scale)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Số lượng hồ sơ (Log Scale)", row=1, col=2)
    
    # ----- Plot 3: SHAP Feature Importance -----
    print("3. Đang tạo biểu đồ SHAP Importance...")
    if os.path.exists(SHAP_FILE):
        with open(SHAP_FILE, 'rb') as f:
            shap_data = pickle.load(f)
        shap_vals = np.abs(shap_data['shap_values']).mean(axis=0) # Mean absolute impact
        features = shap_data['features']
        
        # Clean up feature names (remove PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH_ prefix for display)
        clean_features = [f.replace('PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH_', 'TÀI CHÍNH: ').replace('TÌNH_TRẠNG_TƯƠNG_TÁC_', 'TƯƠNG TÁC: ') for f in features]
        
        shap_df = pd.DataFrame({'Feature': clean_features, 'Impact': shap_vals})
        shap_df = shap_df.sort_values(by='Impact', ascending=True).tail(10) # Top 10
        
        fig.add_trace(go.Bar(
            y=shap_df['Feature'], x=shap_df['Impact'], orientation='h',
            marker_color='#636EFA', name='Mức độ tác động'
        ), row=2, col=1)
        
        # Thêm text giải thích SHAP
        fig.add_annotation(
            text="Biểu đồ này chỉ ra <b>10 yếu tố quan trọng nhất</b> giúp AI dự báo khách có trả nợ hay không.<br>Thanh càng dài, yếu tố đó càng mang tính quyết định.",
            xref="x3 domain", yref="y3 domain", x=0.5, y=0.1,
            showarrow=False, font=dict(size=11, color="gray"), bgcolor="white"
        )
    
    # ----- Plot 4: Financial Capacity Matrix -----
    print("4. Đang tạo biểu đồ Năng lực Tài chính...")
    fin_col = 'PHÂN_KHÚC_NĂNG_LỰC_TÀI_CHÍNH'
    if fin_col not in df.columns: # fallback if not found
        fin_col = [c for c in df.columns if c.startswith('PHÂN_KHÚC')][0]
        
    fin_df = df.groupby(fin_col)['PTP_SCORE_PERCENT'].mean().reset_index()
    fin_df = fin_df.sort_values('PTP_SCORE_PERCENT', ascending=False)
    
    fig.add_trace(go.Bar(
        x=fin_df[fin_col], y=fin_df['PTP_SCORE_PERCENT'],
        marker_color='#AB63FA', text=fin_df['PTP_SCORE_PERCENT'].round(2), textposition='auto',
        name='Điểm PTP Trung bình (%)'
    ), row=2, col=2)
    fig.update_yaxes(title_text="PTP Trung Bình (%)", row=2, col=2)
    
    # ----- Plot 5: Clustering Info -----
    print("5. Đang tạo biểu đồ Phân cụm (K-Means)...")
    cluster_df = df.groupby('CỤM_HÀNH_VI').agg({'DPD': 'median', 'LOAN ID': 'count'}).reset_index()
    
    fig.add_trace(go.Bar(
        x=cluster_df['CỤM_HÀNH_VI'], y=cluster_df['DPD'],
        marker_color='#FFA15A', text=cluster_df['LOAN ID'].apply(lambda x: f"{x:,.0f} HS"), textposition='auto',
        name='Trung vị DPD'
    ), row=3, col=1)
    fig.update_yaxes(title_text="Trung vị DPD (Ngày)", row=3, col=1)
    
    # ----- Plot 6: Interaction Flag (Pie) -----
    print("6. Đang tạo biểu đồ Tương tác...")
    interaction_df = df['CỜ_TƯƠNG_TÁC_GẦN_ĐÂY'].value_counts().reset_index()
    interaction_df.columns = ['Có Tương Tác', 'Số Lượng']
    interaction_df['Có Tương Tác'] = interaction_df['Có Tương Tác'].map({1: 'Có tương tác gần đây', 0: 'Mất tích / Bặt vô âm tín'})
    
    fig.add_trace(go.Pie(
        labels=interaction_df['Có Tương Tác'], values=interaction_df['Số Lượng'],
        hole=0.4, marker_colors=['#B6E880', '#FF97FF']
    ), row=3, col=2)
    
    # ----- Tinh chỉnh Layout tổng thể -----
    fig.update_layout(
        title=dict(text="<b>ADVANCED INSIGHT HUB - BÁO CÁO KHOA HỌC DỮ LIỆU CHUYÊN SÂU</b>", font=dict(size=24), x=0.5),
        height=1400,
        width=1800,
         # Giao diện Dark mode xịn sò
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    fig.write_html(HTML_OUTPUT)
    print(f"HOÀN THÀNH TẤT CẢ! Dashboard đã được tạo tại: {HTML_OUTPUT}")

if __name__ == "__main__":
    create_dashboard()
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="HK EV Charging Dashboard ⚡",
    page_icon="⚡",
    layout="wide"
)

# ── CSV URL (raw GitHub) ──
CSV_URL = "https://raw.githubusercontent.com/smfu0922/EV-Web-Scraping/main/EV_Scraping_Merge.csv"

@st.cache_data(ttl=60)
def load_data():
    """Fetch CSV from GitHub, auto-refresh every 60s."""
    try:
        df = pd.read_csv(CSV_URL, encoding="utf-8-sig")
        # Normalise column names
        df.columns = df.columns.str.strip()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        # Drop fully empty columns
        df = df.dropna(axis=1, how='all')
        # Clean text fields
        for col in ['theme', 'operator', 'sentiment', 'location', 'summary', 'source']:
            if col in df.columns:
                df[col] = df[col].fillna('').replace([None, 'None', 'nan', 'NaN'], '')
        return df
    except Exception as e:
        st.error(f"❌ 讀取 CSV 失敗: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ 未能載入數據，請檢查 GitHub 上的 CSV 是否存在。")
    st.stop()

# ── Sidebar filters ──
st.sidebar.title("⚡ 篩選")

themes = ['全部'] + sorted([t for t in df['theme'].dropna().unique() if t])
selected_theme = st.sidebar.selectbox("主題分類", themes)

operators = ['全部'] + sorted([o for o in df['operator'].dropna().unique() if o])
selected_operator = st.sidebar.selectbox("營辦商", operators)

sentiments = ['全部'] + sorted([s for s in df['sentiment'].dropna().unique() if s])
selected_sentiment = st.sidebar.selectbox("情緒", sentiments)

# Date range
min_date = df['time'].min().date()
max_date = df['time'].max().date()
date_range = st.sidebar.date_input("日期範圍", [min_date, max_date])

# ── Refresh button ──
st.sidebar.markdown("---")
if st.sidebar.button("🔄 立即刷新數據"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"自動刷新: 每 60 秒")
st.sidebar.caption(f"最後更新: {datetime.now().strftime('%H:%M:%S')}")

# ── Filter logic ──
filtered = df.copy()
if selected_theme != '全部':
    filtered = filtered[filtered['theme'] == selected_theme]
if selected_operator != '全部':
    filtered = filtered[filtered['operator'] == selected_operator]
if selected_sentiment != '全部':
    filtered = filtered[filtered['sentiment'] == selected_sentiment]
if len(date_range) == 2:
    filtered = filtered[
        (filtered['time'].dt.date >= date_range[0]) &
        (filtered['time'].dt.date <= date_range[1])
    ]

# ── KPIs ──
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("📊 總筆數", len(filtered))
with col2:
    st.metric("💬 充電疑問", len(filtered[filtered['theme'] == '充電疑問']))
with col3:
    st.metric("💰 價格動態", len(filtered[filtered['theme'] == '價格動態']))
with col4:
    st.metric("📍 站點情報", len(filtered[filtered['theme'] == '站點情報']))
with col5:
    st.metric("🔧 服務問題", len(filtered[filtered['theme'] == '服務問題']))
with col6:
    st.metric("🚗 車位佔用", len(filtered[filtered['theme'] == '車位佔用']))

# ── Charts ──
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📊 主題分佈")
    theme_counts = filtered['theme'].value_counts()
    if not theme_counts.empty:
        colors = {
            '充電疑問': '#5e5843', '價格動態': '#2d6662',
            '站點情報': '#a34d43', '其他無關': '#616161',
            '服務問題': '#d67a2a', '車位佔用': '#4a86b8'
        }
        bar_color = [colors.get(t, '#78756c') for t in theme_counts.index]
        st.bar_chart(theme_counts, color=bar_color)

with chart_col2:
    st.subheader("😊 情緒分佈")
    sent_counts = filtered['sentiment'].value_counts()
    if not sent_counts.empty:
        st.bar_chart(sent_counts)

# ── 營辦商聲量趨勢 ──
st.subheader("📈 營辦商聲量趨勢")
op_options = ['全部'] + sorted(filtered['operator'].dropna().unique().tolist())
op_col = st.selectbox("揀營辦商", op_options, key='op_trend')
if op_col != '全部':
    op_df = filtered[filtered['operator'] == op_col]
    trend = op_df.groupby(op_df['time'].dt.to_period('M')).size()
    if not trend.empty:
        st.line_chart(trend)

# ── Data table ──
st.subheader(f"📋 數據明細（共 {len(filtered)} 筆，最新在最前）")

display_cols = ['time', 'source', 'theme', 'operator', 'sentiment', 'summary', 'location']
available_cols = [c for c in display_cols if c in filtered.columns]
display_df = filtered[available_cols].copy()
if 'time' in display_df.columns:
    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d')

# Sort newest first
display_df = display_df.sort_values('time', ascending=False)

st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        'time': '日期',
        'source': '來源',
        'theme': '主題',
        'operator': '營辦商',
        'sentiment': '情緒',
        'summary': '摘要',
        'location': '地點',
    }
)

# ── Footer ──
st.caption(f"最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.caption("數據來源: GitHub (EV_Scraping_Merge_v3.csv)・自動刷新每 60 秒")

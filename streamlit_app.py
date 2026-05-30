import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(
    page_title="HK EV Charging Dashboard ⚡",
    page_icon="⚡",
    layout="wide"
)

# ── CSV URL (raw GitHub) ──
CSV_URL = "https://raw.githubusercontent.com/smfu0922/EV-Web-Scraping/main/EV_Scraping_Merge.csv"

@st.cache_data(ttl=60)
def load_and_process():
    """Fetch CSV from GitHub and process data (same logic as ev_charging_insight_dashboard_new.py)"""
    df = pd.read_csv(CSV_URL, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Required columns
    required = [
        "time", "source", "operator", "sentiment", "location", "title", "text",
        "shared_post_text", "theme", "charge_type", "landlord", "post_url",
        "summary", "user_impact", "comp_insight"
    ]
    available = [c for c in required if c in df.columns]
    df = df[available]

    # 1.1 Time parsing
    df['parsed_time'] = pd.to_datetime(df['time'], errors='coerce')
    default_date = df['parsed_time'].dropna().max() if not df['parsed_time'].dropna().empty else pd.Timestamp.now()
    df['clean_date'] = df['parsed_time'].fillna(default_date).dt.strftime('%Y-%m-%d')
    df['month_str'] = df['parsed_time'].fillna(default_date).dt.strftime('%Y-%m')

    # 1.2 Text normalisation
    standard_replacements = [np.nan, "nan", "NaN", "None", "none", "null", ""]
    text_cols = ["title", "text", "shared_post_text", "location", "landlord", "post_url", "summary", "user_impact",
                 "comp_insight"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].replace(standard_replacements, "").astype(str).str.strip()

    if 'theme' in df.columns:
        df['theme'] = df['theme'].fillna('其他無關').replace(standard_replacements, "其他無關")

    categorical_defaults = {
        "operator": "Unknown", "charge_type": "中充", "sentiment": "Neutral", "source": "社區論壇"
    }
    for col, default_val in categorical_defaults.items():
        df[col] = df.get(col, pd.Series()).replace(standard_replacements, default_val).astype(str).str.strip()

    months_list = sorted(df['month_str'].dropna().unique())

    # 1.3 Monthly insights
    monthly_insights_db = {}
    for i, m in enumerate(months_list):
        m_data = df[df['month_str'] == m]
        prev_m = months_list[i - 1] if i > 0 else None
        prev_m_data = df[df['month_str'] == prev_m] if prev_m else pd.DataFrame()

        cur_vol = len(m_data)
        prev_vol = len(prev_m_data)

        if prev_vol > 0:
            pct = round(((cur_vol - prev_vol) / prev_vol) * 100)
            pct_str = f"{'+' if pct >= 0 else ''}{pct}%"
        else:
            pct_str = "基準月建立"

        theme_counts = m_data['theme'].value_counts()
        top_theme = theme_counts.index[0] if not theme_counts.empty else "無"

        neg_data = m_data[m_data['sentiment'] == 'Negative']
        neg_theme_str = "無異常集中吐槽點"
        neg_detail_str = "🟢 本月車主反饋良好，全港公共與商業充電基建未見集體性抱怨。"

        if not neg_data.empty:
            neg_theme_counts = neg_data['theme'].value_counts()
            top_neg_theme = neg_theme_counts.index[0]
            neg_theme_str = f"【{top_neg_theme}】主題怨氣集中"
            summaries_list = [s for s in neg_data['summary'].tolist() if s and s != '未提及' and s != '']
            if summaries_list:
                neg_detail_str = "⚠️ 車主核心不滿：" + "；".join(summaries_list[:2])
                if len(neg_detail_str) > 130:
                    neg_detail_str = neg_detail_str[:130] + "..."
            else:
                neg_detail_str = f"⚠️ 車主社群本月對{top_neg_theme}的相關設施投訴有所攀升。"

        ops_cur = set(m_data['operator'].unique()) - {'Unknown'}
        ops_prev = set(prev_m_data['operator'].unique()) - {'Unknown'} if prev_m else set()
        new_ops = ops_cur - ops_prev
        if new_ops:
            mkt_str = f"🎯 發現新插旗訊號！營辦商 【{', '.join(list(new_ops)[:2])}】 本月首度在社群爆發討論。"
        else:
            top_ops = m_data['operator'].value_counts()
            top_ops = top_ops[top_ops.index != 'Unknown']
            mkt_str = f"🤝 充電格局穩健。熱門提及品牌為 【{', '.join(top_ops.index[:2])}】。" if not top_ops.empty else "🤝 本月市佔格局基本穩健。"

        monthly_insights_db[m] = {
            "volume_title": f"總聲量 {cur_vol} 筆 ({pct_str})",
            "volume_count": f"{cur_vol} 筆",
            "volume_badge": pct_str,
            "negative_title": neg_theme_str,
            "negative_text": neg_detail_str,
            "top_theme": top_theme,
            "market_text": mkt_str
        }

    # 1.4 Sentiment trend
    trend_dataset_db = {}
    for m in months_list:
        m_data = df[df['month_str'] == m]
        s_counts = m_data['sentiment'].value_counts()
        trend_dataset_db[m] = {
            "positive": int(s_counts.get('Positive', 0)),
            "neutral": int(s_counts.get('Neutral', 0)),
            "negative": int(s_counts.get('Negative', 0))
        }

    # Build records
    def is_valid_intel(val):
        val_clean = str(val).strip()
        if val_clean in ["", "未提及", "nan", "NaN", "None", "null", "Unknown", "暫無明確", "暫無顯著"]:
            return False
        if len(val_clean) < 4:
            return False
        return True

    clean_records = []
    for idx, row in df.iterrows():
        formatted_display_text = '<div class="space-y-1.5 font-sans text-gray-800 break-words whitespace-normal text-justify">'
        if row["title"] and row["title"] != "未提及" and row["title"] != " ":
            formatted_display_text += f'<span class="block font-bold text-[#5e5843] text-[13px]">【{row["title"]}】</span>'

        main_text = row["text"][:400]
        formatted_display_text += f'<p class="leading-relaxed text-gray-700">{main_text}</p>'

        if row["shared_post_text"] and row["shared_post_text"] != "未提及" and row["shared_post_text"] != "":
            shared_text = row["shared_post_text"][:300]
            formatted_display_text += f"""
            <div class="mt-2 p-2 bg-gray-50/80 border-l-2 border-gray-300 text-gray-500 rounded-r-md text-[11px] leading-relaxed break-words whitespace-normal">
                {shared_text}
            </div>
            """
        formatted_display_text += '</div>'

        has_intel = False
        if is_valid_intel(row["summary"]) or is_valid_intel(row["user_impact"]) or is_valid_intel(row["comp_insight"]):
            has_intel = True

        record = {
            "id": f"#{idx + 1:03d}",
            "time": row["clean_date"],
            "month": row["month_str"],
            "source": row["source"],
            "operator": row["operator"],
            "sentiment": row["sentiment"],
            "location": row["location"],
            "theme": row["theme"],
            "charge_type": row["charge_type"],
            "landlord": row["landlord"] if row["landlord"] != "未提及" else "未知場主",
            "post_url": row["post_url"],
            "text_html": formatted_display_text,
            "summary": row["summary"] if row["summary"] else "未提及詳細摘要。",
            "user_impact": row["user_impact"] if row["user_impact"] != "未提及" else "暫無明確使用者實質影響反饋。",
            "comp_insight": row["comp_insight"] if row["comp_insight"] != "未提及" else "暫無顯著品牌競爭戰略啟示。",
            "has_intel": has_intel
        }
        clean_records.append(record)

    # Operators
    raw_operators = sorted(list(df["operator"].unique()))
    if "The Point" in raw_operators:
        raw_operators.remove("The Point")
        unique_operators = ["The Point"] + raw_operators
    else:
        unique_operators = raw_operators

    min_date = df['clean_date'].min()
    max_date = df['clean_date'].max()

    # Operator monthly matrix
    operators_monthly_full_matrix = {}
    for op in unique_operators:
        if op == 'Unknown' or op == '':
            continue
        operators_monthly_full_matrix[op] = {}
        for m in months_list:
            operators_monthly_full_matrix[op][m] = int(len(df[(df['operator'] == op) & (df['month_str'] == m)]))

    return {
        "records": clean_records,
        "operators": unique_operators,
        "monthly_insights": monthly_insights_db,
        "trend_dataset": trend_dataset_db,
        "months": months_list,
        "operator_matrix": operators_monthly_full_matrix,
        "min_date": min_date,
        "max_date": max_date
    }


# ── Load data ──
data = load_and_process()

json_dataset = json.dumps(data["records"], ensure_ascii=False)
json_operators = json.dumps(data["operators"], ensure_ascii=False)
json_monthly_insights = json.dumps(data["monthly_insights"], ensure_ascii=False)
json_trend_dataset = json.dumps(data["trend_dataset"], ensure_ascii=False)
json_all_months = json.dumps(data["months"], ensure_ascii=False)
json_operators_monthly_matrix = json.dumps(data["operator_matrix"], ensure_ascii=False)
min_date = data["min_date"]
max_date = data["max_date"]

# ── HTML template (same as ev_charging_insight_dashboard_new.py) ──
html_template = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HK EV Charging Insight Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { background-color: #f5f4ed; color: #2d2d2d; font-family: -apple-system, BlinkMacSystemFont, sans-serif; overflow-x: hidden; }
        .glass-card { background: #ffffff; border: 1px solid #dcd7bc; box-shadow: 0 4px 20px rgba(220, 215, 188, 0.12); }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #eeebe0; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #c6bfa2; border-radius: 3px; }
        .filter-badge { background-color: #4a4a4a; color: #ffffff; padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; display: inline-flex; }
        .mom-card { background: linear-gradient(145deg, #ffffff, #fdfdfb); border: 1px solid #dcd7bc; height: 100%; }
        .table-responsive-box { overflow-x: auto; width: 100%; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border-bottom: 1px solid #eeebe0; }
    </style>
</head>
<body class="p-6 antialiased relative">

    <header class="mb-6 p-5 rounded-2xl glass-card flex flex-col md:flex-row justify-between items-start md:items-center border-l-8 border-[#5e5843]">
        <div>
            <h1 class="text-2xl font-bold tracking-tight text-[#2b271d]">🌐 HK EV Charging Premium Business Dashboard</h1>
        </div>
        <div class="flex flex-wrap items-center gap-2 mt-3 md:mt-0 max-w-3xl">
            <div id="statusTheme" class="filter-badge">🎯 主題: 全部</div>
            <div id="statusOperator" class="filter-badge">⚡ 營辦商: 全部</div>
            <div id="statusDate" class="filter-badge">📅 期間: 全部</div>
            <div id="statusLocation" class="filter-badge">📍 地點: 全部</div>
            <div id="totalNumBadge" class="text-sm bg-[#5e5843]/10 text-[#5e5843] px-4 py-1 rounded-full border border-[#5e5843]/30 font-mono font-bold">Total: -- 筆</div>
        </div>
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div class="glass-card p-5 rounded-2xl">
            <h2 class="text-sm font-bold text-gray-700 mb-3 border-l-4 border-[#8c7e5a] pl-2">📊 Bars | 5大主題數值分佈 (點擊長條同步聯動)</h2>
            <div id="barChart" style="width: 100%; height: 260px;"></div>
        </div>
        <div class="glass-card p-5 rounded-2xl">
            <h2 class="text-sm font-bold text-gray-700 mb-3 border-l-4 border-[#8c7e5a] pl-2">🎯 Pies | 主題佔比分佈 (多維穿透)</h2>
            <div id="donutChart" style="width: 100%; height: 260px;"></div>
        </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div class="glass-card p-5 rounded-2xl flex flex-col justify-between h-[360px]">
            <div>
                <h2 class="text-sm font-bold text-gray-700 mb-2 border-l-4 border-[#5e5843] pl-2">🎛️ 智能數據控制台</h2>
                <div class="mb-2">
                    <div class="flex justify-between items-center mb-0.5">
                        <label class="text-[11px] font-semibold text-gray-500">🔍 選擇營辦商 (多選)</label>
                        <button onclick="clearOperatorSelection()" class="text-[10px] text-blue-600 hover:underline">清空</button>
                    </div>
                    <div id="operatorCheckboxContainer" class="bg-[#faf9f5] rounded-lg p-2 border border-[#dcd7bc] h-24 overflow-y-auto custom-scrollbar flex flex-col gap-1"></div>
                </div>
                <div class="mb-2">
                    <label class="block text-[11px] font-semibold text-gray-500 mb-0.5">📅 自訂觀測日期區間</label>
                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <span class="text-[9px] text-gray-400 block">開始日期</span>
                            <input type="date" id="dateStart" class="w-full bg-[#faf9f5] text-[11px] text-gray-700 rounded-md p-1 border border-[#dcd7bc] focus:outline-none">
                        </div>
                        <div>
                            <span class="text-[9px] text-gray-400 block">結束日期</span>
                            <input type="date" id="dateEnd" class="w-full bg-[#faf9f5] text-[11px] text-gray-700 rounded-md p-1 border border-[#dcd7bc] focus:outline-none">
                        </div>
                    </div>
                </div>
            </div>
            <button onclick="resetAllFilters()" class="w-full py-1.5 bg-gray-200 hover:bg-gray-300 text-gray-700 text-[11px] font-bold rounded-lg transition-all border border-gray-300 shadow-sm">🔄 一鍵重置所有篩選</button>
        </div>

        <div class="glass-card p-5 rounded-2xl lg:col-span-2 flex flex-col justify-between h-[360px]">
            <div class="mb-1">
                <h2 class="text-sm font-bold text-gray-700">🗺️ 全港營辦商地理分佈</h2>
                <div id="mapFilterStatus" class="text-[10px] bg-[#8c7e5a]/10 text-[#8c7e5a] px-2 py-0.5 rounded-md mt-1 hidden items-center gap-1 font-medium w-fit">
                    <span id="currentMapLoc"></span>
                    <button onclick="clearMapFilter()" class="text-gray-400 hover:text-red-500 font-bold">×</button>
                </div>
            </div>
            <div id="liveMap" class="rounded-xl h-[270px] border border-[#dcd7bc] z-10"></div>
        </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div class="glass-card p-5 rounded-2xl">
            <h2 class="text-sm font-bold text-gray-700 mb-1 border-l-4 border-blue-600 pl-2">📊 歷史大局觀 | 全港充電營辦商全期社群聲量趨勢矩陣</h2>
            <p class="text-[11px] text-gray-400 mb-2">已剔除 Unknown。支援控制台勾選多選與點擊折線雙向聯動變更曲線！</p>
            <div id="operatorsFullTimelineChart" style="width: 100%; height: 280px;"></div>
        </div>
        <div class="glass-card p-5 rounded-2xl">
            <h2 class="text-sm font-bold text-gray-700 mb-1 border-l-4 border-emerald-600 pl-2">🎭 品牌與輿情 | 各充電營辦商情緒比例堆疊分佈圖</h2>
            <p class="text-[11px] text-gray-400 mb-2">已剔除 Unknown。動態響應全域過濾，滑鼠懸停可觀看絕對發文筆數與精準佔比。</p>
            <div id="operatorSentimentStackChart" style="width: 100%; height: 280px;"></div>
        </div>
    </div>

    <div class="glass-card p-5 rounded-2xl mb-6">
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4 border-b border-gray-100 pb-3">
            <div>
                <h2 class="text-sm font-bold text-gray-700 border-l-4 border-[#8c7e5a] pl-2">📈 MoM 核心商情環比異動監控 (動態智慧解碼)</h2>
                <p class="text-[11px] text-gray-400 mt-0.5">切換觀測月份，解碼當月的輿情焦點事件與情緒滾動線</p>
            </div>
            <div class="flex items-center gap-2">
                <label class="text-xs font-semibold text-gray-500 whitespace-nowrap">觀測月份:</label>
                <select id="monthSelector" onchange="switchMonthInsight()" class="bg-[#faf9f5] text-xs font-bold text-[#5e5843] rounded-xl p-2 border border-[#dcd7bc] focus:outline-none"></select>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="mom-card p-4 rounded-xl flex flex-col justify-between">
                    <div><span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider block">📢 輿情聲量環比</span></div>
                    <div class="flex items-baseline justify-between mt-2">
                        <span id="momVolumeCount" class="text-2xl font-mono font-bold text-slate-800">-- 筆</span>
                        <span id="momVolumeBadge" class="text-xs font-bold px-2 py-0.5 rounded-full">--</span>
                    </div>
                </div>
                <div class="mom-card p-4 rounded-xl flex flex-col justify-between border-t-4 border-t-red-600">
                    <div><span id="momNegativeTitle" class="text-[11px] font-bold text-red-500 uppercase tracking-wider block">⚠️ 負面吐槽焦點</span>
                        <p id="momNegativeText" class="text-xs text-gray-600 mt-2 leading-relaxed font-medium">正在載入數據焦點...</p>
                    </div>
                </div>
                <div class="mom-card p-4 rounded-xl flex flex-col justify-between border-t-4 border-t-indigo-600">
                    <div><span class="text-[11px] font-bold text-indigo-600 uppercase tracking-wider block">🤝 競爭格局訊號</span>
                        <p id="momMarketText" class="text-xs text-gray-600 mt-2 leading-relaxed font-medium">正在對比捕捉...</p>
                    </div>
                </div>
            </div>
            <div class="bg-[#faf9f5] p-3 rounded-xl border border-[#dcd7bc]">
                <span id="trendChartTitle" class="text-[11px] font-bold text-gray-500 uppercase tracking-wider block mb-1">🟢 🟡 🔴 滾動情緒趨勢</span>
                <div id="trendLineChart" style="width: 100%; height: 110px;"></div>
            </div>
        </div>
    </div>

    <section class="glass-card rounded-2xl p-5">
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-4">
            <h2 class="text-sm font-bold text-gray-700 border-l-4 border-[#2d2d2d] pl-2">📋 原始發文與評論 Row Data (高端商情側邊滑出解碼版)</h2>
            <div class="flex items-center gap-2 text-xs">
                <button onclick="changePage(-1)" id="btnPrev" class="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-600 font-semibold rounded-lg disabled:opacity-30">◀ 上一頁</button>
                <span id="pageIndicator" class="font-mono font-medium text-gray-500">Page 1 / 1</span>
                <button onclick="changePage(1)" id="btnNext" class="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-600 font-semibold rounded-lg disabled:opacity-30">下一頁 ▶</button>
            </div>
        </div>
        <div class="table-responsive-box custom-scrollbar">
            <table class="w-full text-left table-layout-auto">
                <thead>
                    <tr class="border-b border-[#dcd7bc] text-xs text-gray-600 bg-[#faf9f5]">
                        <th class="p-3 font-bold w-[65px] whitespace-nowrap">編號</th>
                        <th class="p-3 font-bold w-[85px] whitespace-nowrap">發言日期</th>
                        <th class="p-3 font-bold w-[100px] min-w-[100px] max-w-[90px] break-all whitespace-normal">來源</th>
                        <th class="p-3 font-bold w-[140px]">營辦商 / 場主</th>
                        <th class="p-3 font-bold w-[85px] whitespace-nowrap">分類主題</th>
                        <th class="p-3 font-bold w-[70px] whitespace-nowrap">情緒</th>
                        <th class="p-3 font-bold w-[100px] min-w-[100px] max-w-[100px] break-all whitespace-normal">提及地點</th>
                        <th class="p-3 font-bold w-full min-w-[360px]">留言原文</th>
                        <th class="p-3 font-bold text-center w-[75px] whitespace-nowrap">深度商情</th>
                    </tr>
                </thead>
                <tbody id="dataTableBody" class="text-xs divide-y divide-gray-100 bg-white"></tbody>
            </table>
        </div>
    </section>

    <div id="rightIntelPanel" class="fixed top-0 right-0 h-full w-[380px] sm:w-[420px] bg-white border-l-4 border-[#5e5843] shadow-2xl transform translate-x-full transition-transform duration-300 ease-in-out z-[99999] p-6 flex flex-col justify-between">
        <div>
            <div class="flex justify-between items-center border-b border-gray-200 pb-3 mb-5">
                <h3 class="font-bold text-gray-800 text-base flex items-center gap-2">📋 商業智庫情報解碼 <span id="panelRowId" class="text-xs text-gray-400 font-mono font-normal"></span></h3>
                <button onclick="closeIntelPanel()" class="text-gray-400 hover:text-gray-600 text-xl font-bold p-1">✕</button>
            </div>
            <div class="space-y-5 overflow-y-auto custom-scrollbar pr-1" style="max-height: calc(100vh - 140px);">
                <div><b class="text-[#8c7e5a] text-xs uppercase tracking-wider block mb-1">📝 核心摘要 (Summary)</b>
                    <p id="panelSummary" class="text-xs text-gray-600 bg-gray-50 p-3 rounded-lg font-medium border border-gray-100 leading-relaxed break-words text-justify"></p>
                </div>
                <div><b class="text-amber-700 text-xs uppercase tracking-wider block mb-1">👥 車主實質影響 (User Impact)</b>
                    <p id="panelUserImpact" class="text-xs text-gray-700 bg-amber-50/10 p-3 rounded-lg font-medium border border-amber-100/70 leading-relaxed break-words text-justify"></p>
                </div>
                <div><b class="text-emerald-700 text-xs uppercase tracking-wider block mb-1">🎯 競爭戰略啟示 (Comp Insight)</b>
                    <p id="panelCompInsight" class="text-xs text-gray-700 bg-emerald-50/10 p-3 rounded-lg font-medium border border-emerald-100/70 leading-relaxed break-words text-justify"></p>
                </div>
            </div>
        </div>
        <button onclick="closeIntelPanel()" class="w-full py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-bold rounded-xl transition-colors border border-gray-200 shadow-sm">關閉商情面板</button>
    </div>

    <script>
        const rawDataset = __RAW_DATASET__;
        const allOperators = __ALL_OPERATORS__;
        const monthlyInsights = __MONTHLY_INSIGHTS__;
        const trendDataset = __TREND_DATASET__;
        const allMonthsList = __ALL_MONTHS__;
        const operatorsMonthlyMatrix = __OPERATORS_MONTHLY_MATRIX__;
        const absoluteMinDate = '__MIN_DATE__';
        const absoluteMaxDate = '__MAX_DATE__';

        const themeColors = { "充電疑問": "#5e5843", "價格動態": "#2d6662", "站點情報": "#a34d43", "其他無關": "#616161", "服務問題": "#d67a2a", "車位佔用": "#4a86b8" };
        const fallbackMapColor = "#78756c";

        const locationCoords = {
            "尖沙咀": [22.2988, 114.1722], "中環": [22.2819, 114.1580], "銅鑼灣": [22.2860, 114.1850],
            "旺角": [22.3204, 114.1698], "觀塘": [22.3104, 114.2231], "沙田": [22.3832, 114.1879],
            "荃灣": [22.3686, 114.1131], "元朗": [22.4456, 114.0222], "將軍澳": [22.3121, 114.2589],
            "屯門": [22.3964, 113.9743], "九龍灣": [22.3225, 114.2115], "紅磡": [22.3020, 114.1843],
            "大角咀": [22.3218, 114.1601], "深水埗": [22.3286, 114.1603], "金鐘": [22.2783, 114.1645],
            "東涌": [22.2882, 113.9422], "火炭": [22.3956, 114.1953], "啟德": [22.3222, 114.2056]
        };

        let selectedOperators = []; let clickedTheme = null; let clickedLocation = null;
        let currentPage = 1; const pageSize = 20; let currentlyFilteredData = [];

        let map = null; let markerGroup = null;
        let barChart = echarts.init(document.getElementById('barChart'));
        let donutChart = echarts.init(document.getElementById('donutChart'));
        let trendLineChart = echarts.init(document.getElementById('trendLineChart'));
        let operatorsFullTimelineChart = echarts.init(document.getElementById('operatorsFullTimelineChart'));
        let operatorSentimentStackChart = echarts.init(document.getElementById('operatorSentimentStackChart'));

        window.addEventListener('DOMContentLoaded', () => {
            const startInput = document.getElementById('dateStart');
            const endInput = document.getElementById('dateEnd');
            startInput.value = absoluteMinDate; startInput.min = absoluteMinDate; startInput.max = absoluteMaxDate;
            endInput.value = absoluteMaxDate; endInput.min = absoluteMinDate; endInput.max = absoluteMaxDate;
            startInput.addEventListener('change', () => { currentPage = 1; renderDashboard(); });
            endInput.addEventListener('change', () => { currentPage = 1; renderDashboard(); });
            initLeafletMap();
            initMonthSelector();
            renderOperatorCheckboxes();
            renderDashboard();
            barChart.on('click', (p) => { if (p.componentType === 'series') { clickedTheme = (clickedTheme === p.name) ? null : p.name; currentPage = 1; renderDashboard(); } });
            donutChart.on('click', (p) => { if (p.componentType === 'series') { clickedTheme = (clickedTheme === p.name) ? null : p.name; currentPage = 1; renderDashboard(); } });
            operatorsFullTimelineChart.on('click', (p) => {
                if (p.componentType === 'series') {
                    const opName = p.seriesName;
                    const cb = document.getElementById(`op_${opName}`);
                    if (cb) { cb.checked = !cb.checked; handleOperatorChange(); }
                }
            });
            window.addEventListener('resize', () => { barChart.resize(); donutChart.resize(); trendLineChart.resize(); operatorsFullTimelineChart.resize(); operatorSentimentStackChart.resize(); });
        });

        function initLeafletMap() {
            map = L.map('liveMap', { zoomControl: false }).setView([22.3193, 114.1694], 11);
            L.control.zoom({ position: 'bottomright' }).addTo(map);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png').addTo(map);
            markerGroup = L.layerGroup().addTo(map);
        }

        function createMapIcon(color, isSelected) {
            const size = isSelected ? 35 : 26;
            const strokeColor = isSelected ? "#000000" : "#ffffff";
            const strokeW = isSelected ? 3 : 1.5;
            const html = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"><path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" fill="${color}" stroke="${strokeColor}" stroke-width="${strokeW}"/><path d="M11.5 6L8.5 11H12.5L11.5 16L15.5 10H11.5L11.5 6Z" fill="white"/></svg>`;
            return L.divIcon({ html: html, className: '', iconSize: [size, size], iconAnchor: [size/2, size] });
        }

        function initMonthSelector() {
            const selector = document.getElementById('monthSelector');
            const months = Object.keys(monthlyInsights).sort().reverse();
            months.forEach(m => { const opt = document.createElement('option'); opt.value = m; opt.textContent = m; selector.appendChild(opt); });
            switchMonthInsight();
        }

        function switchMonthInsight() {
            const currentMonth = document.getElementById('monthSelector').value;
            const info = monthlyInsights[currentMonth];
            if (!info) return;
            document.getElementById('momVolumeCount').textContent = info.volume_count;
            const vBadge = document.getElementById('momVolumeBadge');
            vBadge.textContent = info.volume_badge;
            vBadge.className = info.volume_badge.includes('-') ? "bg-red-50 text-red-700 border border-red-200 px-2 py-0.5 rounded-full" : "bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded-full";
            document.getElementById('momNegativeTitle').textContent = info.negative_title;
            document.getElementById('momNegativeText').innerHTML = info.negative_text;
            document.getElementById('momMarketText').innerHTML = info.market_text;
            const allMonthsSorted = Object.keys(monthlyInsights).sort();
            const currIdx = allMonthsSorted.indexOf(currentMonth);
            const startIdx = Math.max(0, currIdx - 5);
            const rollingMonths = allMonthsSorted.slice(startIdx, currIdx + 1);
            document.getElementById('trendChartTitle').textContent = `📈 情緒滾動趨勢 (${rollingMonths[0]}~${currentMonth})`;
            let lineXData = [], posData = [], neuData = [], negData = [];
            rollingMonths.forEach(m => {
                lineXData.push(m);
                let dataM = trendDataset[m] || {positive:0, neutral:0, negative:0};
                posData.push(dataM.positive);
                neuData.push(dataM.neutral);
                negData.push(dataM.negative);
            });
            trendLineChart.setOption({
                grid: { top: 15, bottom: 20, left: 25, right: 10 },
                xAxis: { type: 'category', data: lineXData, axisLabel: { fontSize: 9 } },
                yAxis: { type: 'value', splitLine: { show: false }, axisLabel: { fontSize: 8 } },
                tooltip: { trigger: 'axis' },
                series: [
                    { name: 'Positive', type: 'line', data: posData, smooth: true, itemStyle: { color: '#16a34a' } },
                    { name: 'Neutral', type: 'line', data: neuData, smooth: true, itemStyle: { color: '#ca8a04' } },
                    { name: 'Negative', type: 'line', data: negData, smooth: true, itemStyle: { color: '#dc2626' } }
                ]
            }, true);
        }

        function updateOperatorsTimelineChart(activeOperators) {
            const seriesList = [];
            let targetOps = activeOperators.length > 0 ? activeOperators : Object.keys(operatorsMonthlyMatrix).slice(0, 10);
            targetOps.forEach(op => {
                if (!operatorsMonthlyMatrix[op]) return;
                const dataArr = [];
                allMonthsList.forEach(m => { dataArr.push(operatorsMonthlyMatrix[op][m] || 0); });
                seriesList.push({ name: op, type: 'line', data: dataArr, smooth: true, symbol: 'circle', symbolSize: 5, lineStyle: { width: op === 'The Point' ? 3.5 : 2 } });
            });
            operatorsFullTimelineChart.setOption({
                tooltip: { trigger: 'axis' },
                legend: { type: 'scroll', top: 0, textStyle: { fontSize: 10 } },
                grid: { top: 45, bottom: 25, left: '3%', right: '3%', containLabel: true },
                xAxis: { type: 'category', data: allMonthsList, axisLabel: { fontSize: 10 } },
                yAxis: { type: 'value', minInterval: 1, splitLine: { lineStyle: { color: '#eae6d8' } } },
                series: seriesList
            }, true);
        }

        function renderOperatorCheckboxes() {
            const container = document.getElementById('operatorCheckboxContainer');
            container.innerHTML = '';
            allOperators.forEach(op => {
                const isVIP = (op === "The Point");
                const div = document.createElement('div');
                div.className = "flex items-center gap-2 text-xs text-gray-700";
                div.innerHTML = `<input type="checkbox" id="op_${op}" value="${op}" class="op-checkbox w-4 h-4 rounded text-[#5e5843]" onchange="handleOperatorChange()"><label id="label_op_${op}" for="op_${op}" class="cursor-pointer ${isVIP ? 'font-bold text-slate-900' : 'font-medium'}">${isVIP ? '⭐ ' : ''}${op} <span class="text-[10px] text-gray-400 font-mono" id="count_op_${op}">(0)</span></label>`;
                container.appendChild(div);
            });
        }

        function handleOperatorChange() { selectedOperators = Array.from(document.querySelectorAll('.op-checkbox')).filter(cb => cb.checked).map(cb => cb.value); currentPage = 1; renderDashboard(); }
        function clearOperatorSelection() { document.querySelectorAll('.op-checkbox').forEach(cb => cb.checked = false); selectedOperators = []; currentPage = 1; renderDashboard(); }
        function resetAllFilters() { clearOperatorSelection(); clickedTheme = null; clickedLocation = null; document.getElementById('dateStart').value = absoluteMinDate; document.getElementById('dateEnd').value = absoluteMaxDate; currentPage = 1; renderDashboard(); }
        function clearMapFilter() { clickedLocation = null; currentPage = 1; renderDashboard(); }

        function renderDashboard() {
            const startDate = document.getElementById('dateStart').value || absoluteMinDate;
            const endDate = document.getElementById('dateEnd').value || absoluteMaxDate;
            currentlyFilteredData = rawDataset.filter(item => {
                let matchLoc = true;
                if (clickedLocation) { matchLoc = item.location.includes(clickedLocation); }
                return ((selectedOperators.length === 0) ? true : selectedOperators.includes(item.operator)) &&
                       (clickedTheme ? (item.theme === clickedTheme) : true) &&
                       matchLoc && (item.time >= startDate && item.time <= endDate);
            });
            document.getElementById('totalNumBadge').textContent = `Total: ${currentlyFilteredData.length} 筆`;
            document.getElementById('statusTheme').textContent = clickedTheme ? `🎯 主題: ${clickedTheme}` : "🎯 主題: 全部";
            document.getElementById('statusOperator').textContent = (selectedOperators.length > 0) ? `⚡ 營辦商: ${selectedOperators.join(', ')}` : "⚡ 營辦商: 全部";
            document.getElementById('statusDate').textContent = `📅 期間: ${startDate} 至 ${endDate}`;
            document.getElementById('statusLocation').textContent = clickedLocation ? `📍 地點: ${clickedLocation}` : "📍 地點: 全部";
            updateConsoleIntelligence(startDate, endDate);
            updateCharts(currentlyFilteredData);
            updateSandboxMap(currentlyFilteredData);
            updateOperatorSentimentChart(currentlyFilteredData);
            updateOperatorsTimelineChart(selectedOperators);
            updateTablePage();
        }

        function updateConsoleIntelligence(start, end) {
            allOperators.forEach(op => {
                let count = rawDataset.filter(item => (clickedTheme ? (item.theme === clickedTheme) : true) && item.time >= start && item.time <= end && item.operator === op).length;
                const countSpan = document.getElementById(`count_op_${op}`);
                if (countSpan) countSpan.textContent = `(${count})`;
            });
        }

        function updateCharts(data) {
            let counts = { "充電疑問": 0, "價格動態": 0, "站點情報": 0, "其他無關": 0, "服務問題": 0, "車位佔用": 0 };
            data.forEach(item => { if (counts[item.theme] !== undefined) counts[item.theme]++; });
            const categories = Object.keys(counts);
            barChart.setOption({
                backgroundColor: 'transparent', tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                grid: { left: '3%', right: '12%', bottom: '3%', top: '5%', containLabel: true },
                xAxis: { type: 'value' },
                yAxis: { type: 'category', data: categories },
                series: [{ type: 'bar', data: categories.map(cat => ({ value: counts[cat], itemStyle: { color: (clickedTheme && clickedTheme !== cat) ? '#e0dcce' : themeColors[cat], borderRadius: [0, 4, 4, 0] } })), label: { show: true, position: 'right', formatter: '{c} 筆' } }]
            }, true);
            donutChart.setOption({
                backgroundColor: 'transparent', tooltip: { trigger: 'item' },
                series: [{ type: 'pie', radius: ['42%', '70%'], center: ['40%', '50%'], avoidLabelOverlap: false, label: { show: false }, data: categories.map(cat => ({ name: cat, value: counts[cat], itemStyle: { color: (clickedTheme && clickedTheme !== cat) ? '#e0dcce' : themeColors[cat] } })) }]
            }, true);
        }

        function updateSandboxMap(data) {
            if (!map || !markerGroup) return;
            markerGroup.clearLayers();
            const grouped = {};
            data.forEach(item => {
                let mappedLoc = null;
                Object.keys(locationCoords).forEach(k => { if(item.location.includes(k)) mappedLoc = k; });
                if (mappedLoc) { if (!grouped[mappedLoc]) grouped[mappedLoc] = { total: 0, theme: item.theme }; grouped[mappedLoc].total++; }
            });
            Object.entries(grouped).forEach(([locName, info]) => {
                const marker = L.marker(locationCoords[locName], { icon: createMapIcon(themeColors[info.theme] || fallbackMapColor, (clickedLocation === locName)) });
                marker.bindTooltip(`<b>${locName}</b>: ${info.total} 筆`, { direction: 'top', offset: [0, -10] });
                marker.on('click', () => { clickedLocation = (clickedLocation === locName) ? null : locName; currentPage = 1; renderDashboard(); });
                markerGroup.addLayer(marker);
            });
        }

        function updateOperatorSentimentChart(data) {
            let validRecords = data.filter(item => item.operator !== 'Unknown' && item.operator !== '');
            let opMap = {};
            validRecords.forEach(item => {
                let op = item.operator;
                if (!opMap[op]) opMap[op] = { positive: 0, neutral: 0, negative: 0, total: 0 };
                if (item.sentiment === 'Positive') opMap[op].positive++;
                else if (item.sentiment === 'Negative') opMap[op].negative++;
                else opMap[op].neutral++;
                opMap[op].total++;
            });
            let sortedOps = Object.keys(opMap).sort((a, b) => opMap[b].total - opMap[a].total);
            let displayOps = sortedOps.slice(0, 15); displayOps.reverse();
            let categories = []; let posSeries = []; let neuSeries = []; let negSeries = [];
            displayOps.forEach(op => { categories.push(op); posSeries.push(opMap[op].positive); neuSeries.push(opMap[op].neutral); negSeries.push(opMap[op].negative); });
            operatorSentimentStackChart.setOption({
                backgroundColor: 'transparent',
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: function (params) { let opName = params[0].name; let total = 0; let res = `<b>${opName}</b><br/>`; params.forEach(p => { total += p.value; }); params.forEach(p => { let percent = total > 0 ? ((p.value / total) * 100).toFixed(1) : 0; res += `${p.marker} ${p.seriesName}: <b>${p.value} 筆</b> (${percent}%)<br/>`; }); res += `總提及量: <b>${total} 筆</b>`; return res; } },
                legend: { data: ['正向', '中性', '負向'], top: 0, textStyle: { fontSize: 10 } },
                grid: { left: '3%', right: '10%', bottom: '3%', top: '12%', containLabel: true },
                xAxis: { type: 'value', minInterval: 1 },
                yAxis: { type: 'category', data: categories, axisLabel: { fontSize: 10 } },
                series: [
                    { name: '正向', type: 'bar', stack: 'total', data: posSeries, itemStyle: { color: '#16a34a' } },
                    { name: '中性', type: 'bar', stack: 'total', data: neuSeries, itemStyle: { color: '#ca8a04' } },
                    { name: '負向', type: 'bar', stack: 'total', data: negSeries, itemStyle: { color: '#dc2626' } }
                ]
            }, true);
        }

        function updateTablePage() {
            const tbody = document.getElementById('dataTableBody'); tbody.innerHTML = '';
            const totalPages = Math.ceil(currentlyFilteredData.length / pageSize) || 1;
            if (currentPage > totalPages) currentPage = totalPages;
            document.getElementById('pageIndicator').textContent = `Page ${currentPage} / ${totalPages}`;
            document.getElementById('btnPrev').disabled = (currentPage === 1);
            document.getElementById('btnNext').disabled = (currentPage === totalPages);
            const pageData = currentlyFilteredData.slice((currentPage - 1) * pageSize, currentPage * pageSize);
            if (pageData.length === 0) { tbody.innerHTML = `<tr><td colspan="9" class="p-8 text-center text-gray-400 font-medium">⚠️ 沒有符合篩選條件的數據</td></tr>`; return; }
            const fragment = document.createDocumentFragment();
            pageData.forEach(item => {
                const tr = document.createElement('tr'); tr.className = "hover:bg-[#faf9f5] align-top transition-colors border-b border-gray-100";
                let sentBadge = "";
                if (item.sentiment === "Positive") sentBadge = `<span class="bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded-full font-bold text-[10px]">🟢 正向</span>`;
                else if (item.sentiment === "Negative") sentBadge = `<span class="bg-red-50 text-red-600 border border-red-200 px-2 py-0.5 rounded-full font-bold text-[10px]">🔴 負向</span>`;
                else sentBadge = `<span class="bg-yellow-50 text-yellow-700 border border-yellow-200 px-2 py-0.5 rounded-full font-bold text-[10px]">🟡 中性</span>`;
                let themeBadgeColor = "bg-gray-100 text-gray-700 border-gray-300";
                if (item.theme === "充電疑問") themeBadgeColor = "bg-stone-50 text-stone-700 border-stone-200";
                else if (item.theme === "價格動態") themeBadgeColor = "bg-teal-50 text-teal-700 border-teal-200";
                else if (item.theme === "站點情報") themeBadgeColor = "bg-rose-50 text-rose-700 border-rose-200";
                else if (item.theme === "服務問題") themeBadgeColor = "bg-amber-50 text-amber-700 border-amber-200";
                else if (item.theme === "車位佔用") themeBadgeColor = "bg-blue-50 text-blue-700 border-blue-200";
                let intelButton = '<span class="text-gray-300 font-serif font-light text-center block">-</span>';
                if (item.has_intel) {
                    const safeSummary = item.summary.replace(/'/g, "\\\\'").replace(/"/g, '\\\\"');
                    const safeImpact = item.user_impact.replace(/'/g, "\\\\'").replace(/"/g, '\\\\"');
                    const safeInsight = item.comp_insight.replace(/'/g, "\\\\'").replace(/"/g, '\\\\"');
                    intelButton = `<button onclick="openIntelPanel('${item.id}', '${safeSummary}', '${safeImpact}', '${safeInsight}')" class="bg-amber-100 hover:bg-amber-200 text-amber-800 font-bold px-2.5 py-1 rounded-lg flex items-center gap-1 transition-all border border-amber-300 shadow-sm mx-auto animate-pulse"><span>💡</span> <span class="text-[10px]">解碼</span></button>`;
                }
                tr.innerHTML = `
                    <td class="p-3 text-gray-400 font-mono font-medium whitespace-nowrap"><a href="${item.post_url}" target="_blank" class="text-blue-600 hover:underline font-bold">${item.id} 🔗</a></td>
                    <td class="p-3 text-gray-500 font-mono whitespace-nowrap">${item.time}</td>
                    <td class="p-3 text-gray-400 font-medium w-[100px] min-w-[100px] max-w-[100px] break-all whitespace-normal">${item.source}</td>
                    <td class="p-3 font-medium text-slate-700 max-w-[140px] break-words"><div class="font-bold">${item.operator}</div><div class="text-[10px] text-gray-400 mt-0.5">場主: ${item.landlord}</div></td>
                    <td class="p-3 whitespace-nowrap"><span class="${themeBadgeColor} border px-2 py-0.5 rounded-md font-bold text-[10px]">${item.theme}</span></td>
                    <td class="p-3 whitespace-nowrap">${sentBadge}</td>
                    <td class="p-3 text-gray-600 font-bold w-[100px] min-w-[100px] max-w-[100px] break-all whitespace-normal">${item.location || '<span class="text-gray-300 font-normal">Unknown</span>'}</td>
                    <td class="p-3 break-words text-justify whitespace-normal w-full min-w-[360px]">${item.text_html}</td>
                    <td class="p-3 text-center vertical-align-middle whitespace-nowrap">${intelButton}</td>
                `;
                fragment.appendChild(tr);
            });
            tbody.appendChild(fragment);
        }

        function openIntelPanel(id, summary, userImpact, compInsight) {
            document.getElementById('panelRowId').textContent = id;
            document.getElementById('panelSummary').textContent = summary;
            document.getElementById('panelUserImpact').textContent = userImpact;
            document.getElementById('panelCompInsight').textContent = compInsight;
            const panel = document.getElementById('rightIntelPanel');
            panel.classList.remove('translate-x-full'); panel.classList.add('translate-x-0');
        }

        function closeIntelPanel() {
            const panel = document.getElementById('rightIntelPanel');
            panel.classList.remove('translate-x-0'); panel.classList.add('translate-x-full');
        }

        function changePage(direction) { currentPage += direction; updateTablePage(); }
    </script>
</body>
</html>"""

# ── Inject data into template ──
output_html = html_template.replace("__RAW_DATASET__", json_dataset)
output_html = output_html.replace("__ALL_OPERATORS__", json_operators)
output_html = output_html.replace("__MONTHLY_INSIGHTS__", json_monthly_insights)
output_html = output_html.replace("__TREND_DATASET__", json_trend_dataset)
output_html = output_html.replace("__ALL_MONTHS__", json_all_months)
output_html = output_html.replace("__OPERATORS_MONTHLY_MATRIX__", json_operators_monthly_matrix)
output_html = output_html.replace("__MIN_DATE__", min_date)
output_html = output_html.replace("__MAX_DATE__", max_date)

# ── Render HTML in Streamlit ──
st.components.v1.html(output_html, height=2000, scrolling=True)

st.caption(f"數據來源: GitHub (EV_Scraping_Merge.csv)・自動刷新每 60 秒")

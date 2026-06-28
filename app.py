import os
import json
from datetime import datetime

import streamlit as st
from briefing_graph import run_briefing

PORTFOLIO_FILE = "portfolio.json"
DEFAULT_PORTFOLIO = [
    {"name": "SK하이닉스", "ticker": "000660.KS"},
    {"name": "삼성전자", "ticker": "005930.KS"},
    {"name": "마이크론", "ticker": "MU"},
]


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_PORTFOLIO
    return DEFAULT_PORTFOLIO


def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


st.set_page_config(page_title="AI 투자 브리핑", page_icon="📈", layout="wide")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

# ---------- 사이드바 ----------
st.sidebar.header("💼 내 포트폴리오")
st.sidebar.caption("저장해두면 매번 입력할 필요 없어요. 바뀔 때만 수정하세요.")

current = "\n".join(f"{p['name']},{p['ticker']}" for p in st.session_state.portfolio)
edited = st.sidebar.text_area("종목명,티커 (한 줄에 하나)", current, height=180)

c1, c2 = st.sidebar.columns(2)
if c1.button("💾 저장", use_container_width=True):
    new = []
    for line in edited.strip().splitlines():
        if "," in line:
            n, t = line.split(",", 1)
            new.append({"name": n.strip(), "ticker": t.strip()})
    if new:
        st.session_state.portfolio = new
        save_portfolio(new)
        st.sidebar.success("저장 완료!")
    else:
        st.sidebar.error("형식 오류")
if c2.button("↩️ 기본값", use_container_width=True):
    st.session_state.portfolio = DEFAULT_PORTFOLIO
    save_portfolio(DEFAULT_PORTFOLIO)
    st.sidebar.info("기본값 복원")

st.sidebar.divider()
st.sidebar.write("**현재 보유 종목**")
for p in st.session_state.portfolio:
    st.sidebar.write(f"- {p['name']} (`{p['ticker']}`)")

# ---------- 본문 ----------
st.title("📈 AI 주식 투자 브리핑 에이전트")
st.caption(f"오늘 날짜 · {datetime.now():%Y-%m-%d (%A)}")

if st.button("🚀 오늘의 브리핑 생성", type="primary", use_container_width=True):
    with st.spinner("데이터 수집 → 뉴스 분석 → 조언 생성 중..."):
        result = run_briefing(st.session_state.portfolio)

    st.subheader("📊 주요 시장 지수")
    cols = st.columns(len(result["indices"]))
    for col, r in zip(cols, result["indices"]):
        if r["price"] is not None:
            col.metric(r["name"], f"{r['price']:,.2f}", f"{r['rate']:+.2f}%")
        else:
            col.metric(r["name"], "N/A", "수집 실패")

    st.subheader("💼 내 보유 종목")
    cols = st.columns(len(result["prices"]))
    for col, r in zip(cols, result["prices"]):
        if r["price"] is not None:
            col.metric(r["name"], f"{r['price']:,.2f}", f"{r['rate']:+.2f}%")
        else:
            col.metric(r["name"], "N/A", "수집 실패")

    t1, t2, t3 = st.tabs(["📰 전체 시황 뉴스", "📰 내 종목 뉴스", "💡 종목별 조언"])
    with t1:
        st.markdown(result["market_news"])
    with t2:
        st.markdown(result["stock_news"])
    with t3:
        st.markdown(result["advice"])
else:
    st.info("⬅️ 왼쪽에서 포트폴리오를 확인하고, 위 **'오늘의 브리핑 생성'** 버튼을 눌러주세요.")

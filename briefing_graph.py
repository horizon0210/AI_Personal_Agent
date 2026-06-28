import os
import time
from datetime import datetime
from typing import TypedDict, List

import yfinance as yf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ddgs import DDGS
from langgraph.graph import StateGraph, START, END

load_dotenv()

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

INDICES = {"S&P 500": "^GSPC", "나스닥": "^IXIC", "코스피": "^KS11"}

ENGLISH_NAMES = {
    "삼성전자": "Samsung Electronics",
    "SK하이닉스": "SK Hynix",
    "현대차": "Hyundai Motor",
    "카카오": "Kakao",
    "네이버": "Naver",
}


# ==========================================
# 1) State
# ==========================================
class BriefingState(TypedDict):
    portfolio: List[dict]
    today: str
    indices: List[dict]
    prices: List[dict]
    market_news: str
    stock_news: str
    signal: str           # Signal Agent
    portfolio_analysis: str  # Portfolio Agent
    action_plan: str      # Advisor Agent
    report: str


# ==========================================
# 공용 헬퍼
# ==========================================
def fetch_quote(ticker):
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if len(data) >= 2:
            today = data["Close"].iloc[-1]
            prev = data["Close"].iloc[-2]
            rate = (today - prev) / prev * 100
            return float(today), float(rate)
    except Exception:
        pass
    return None, None


def rows_to_text(rows, with_ticker=False):
    lines = []
    for r in rows:
        if r["price"] is None:
            lines.append(f"- {r['name']}: 데이터 수집 실패")
        else:
            sign = "+" if r["rate"] > 0 else ""
            label = f"{r['name']}({r['ticker']})" if with_ticker else r["name"]
            lines.append(f"- {label}: {r['price']:,.2f} ({sign}{r['rate']:.2f}%)")
    return "\n".join(lines)


def safe_news(query, max_results=3, retries=3):
    for _ in range(retries):
        try:
            results = list(DDGS().news(query, region="kr-kr", max_results=max_results))
            if results:
                return results
        except Exception:
            pass
        time.sleep(2)
    return []


# ==========================================
# 2) 데이터 수집 노드
# ==========================================
def collect_indices_node(state: BriefingState) -> dict:
    rows = []
    for name, ticker in INDICES.items():
        price, rate = fetch_quote(ticker)
        rows.append({"name": name, "ticker": ticker, "price": price, "rate": rate})
    return {"indices": rows}


def collect_portfolio_node(state: BriefingState) -> dict:
    rows = []
    for item in state["portfolio"]:
        price, rate = fetch_quote(item["ticker"])
        rows.append({"name": item["name"], "ticker": item["ticker"],
                     "price": price, "rate": rate})
    return {"prices": rows}


def collect_market_news_node(state: BriefingState) -> dict:
    out = ""
    for q in ["증시 시황 전망", "Fed 금리", "글로벌 증시"]:
        for r in safe_news(q, 2):
            out += f"🔹 **[{r['title']}]({r.get('url','')})**\n\n   {r.get('body','')}\n\n"
        time.sleep(1)
    return {"market_news": out or "(전체 시황 뉴스를 불러오지 못했습니다)"}


def collect_stock_news_node(state: BriefingState) -> dict:
    out = ""
    for item in state["portfolio"]:
        out += f"### ■ {item['name']}\n\n"
        eng_name = ENGLISH_NAMES.get(item['name'], item['name'])
        fallback_queries = [
            f"{item['name']} 주가",
            f"{item['name']} 주식",
            f"{item['name']}",
            f"{eng_name} stock",
        ]
        results = []
        for query in fallback_queries:
            results = safe_news(query, 2)
            if results:
                break
            time.sleep(2)
        if results:
            for r in results:
                out += f"🔹 **[{r['title']}]({r.get('url','')})**\n\n   {r.get('body','')}\n\n"
        else:
            out += "(뉴스를 불러오지 못했습니다)\n\n"
        time.sleep(2)
    return {"stock_news": out}


# ==========================================
# 3) AI 에이전트 노드
# ==========================================
def signal_agent_node(state: BriefingState) -> dict:
    """📡 Signal Agent — 종목별 긍정/부정 신호 평가"""
    idx_text = rows_to_text(state["indices"])
    port_text = rows_to_text(state["prices"], with_ticker=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    prompt = f"""너는 퀀트 트레이더 출신의 시그널 분석 전문가다.
아래 데이터를 바탕으로 각 종목의 매매 신호를 분석하라.

[출력 형식 — 반드시 지킬 것]
각 종목마다:
### 🟢/🟡/🔴 종목명 — [강한매수 / 매수 / 중립 / 매도 / 강한매도]

**신호 근거**
- 기술적: 오늘 등락률과 시장 대비 강/약 여부
- 뉴스 센티먼트: 긍정/부정 뉴스 비율과 핵심 이슈
- 매크로: 금리/환율/글로벌 증시가 이 종목에 미치는 영향

**신호 강도**: ★★★★☆ (5점 만점)

[규칙]
- 중립 판정은 최대 1개 종목만 허용. 나머지는 반드시 방향성을 제시하라.
- 단정적이고 명확하게 작성하라. 애매한 표현 금지.
- 한국어로 작성

[주요 지수]
{idx_text}

[내 보유 종목]
{port_text}

[전체 시황 뉴스]
{state['market_news'][:1200]}

[보유 종목 뉴스]
{state['stock_news'][:1200]}
"""
    try:
        signal = llm.invoke(prompt).content
    except Exception as e:
        signal = f"(시그널 분석 실패: {e})"
    return {"signal": signal}


def portfolio_agent_node(state: BriefingState) -> dict:
    """📊 Portfolio Agent — 내 포트폴리오 영향 분석"""
    port_text = rows_to_text(state["prices"], with_ticker=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    prompt = f"""너는 포트폴리오 리스크 매니저다.
보유 종목과 시그널 분석을 바탕으로 포트폴리오 전체 관점에서 영향을 분석하라.

[출력 형식]
## 📊 포트폴리오 리스크 스코어: X/10

### 산업 집중도 분석
현재 포트폴리오의 섹터 편중 여부를 구체적으로 지적

### 종목 간 상관관계
종목들이 같은 방향으로 움직이는지, 분산이 되어 있는지 분석

### 현재 시장 환경에서의 취약점
지금 시장 상황에서 이 포트폴리오가 가장 크게 타격받을 시나리오

### 헤지 관점 제안
리스크를 줄이기 위해 추가하면 좋을 섹터/자산 유형 (구체적으로)

[규칙]
- 리스크 스코어는 1~10으로 명확히 수치화 (10이 최고 위험)
- 긍정적인 말로 포장하지 말고 취약점을 직접적으로 지적하라
- 한국어로 작성

[내 보유 종목]
{port_text}

[시그널 분석 결과]
{state['signal'][:1000]}

[전체 시황]
{state['market_news'][:800]}
"""
    try:
        portfolio_analysis = llm.invoke(prompt).content
    except Exception as e:
        portfolio_analysis = f"(포트폴리오 분석 실패: {e})"
    return {"portfolio_analysis": portfolio_analysis}


def advisor_agent_node(state: BriefingState) -> dict:
    """⚡ Advisor Agent — 구체적 액션 플랜 생성"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    prompt = f"""너는 헤지펀드 출신의 과감한 투자 어드바이저다.
시그널 분석과 포트폴리오 분석을 종합해 오늘 당장 실행 가능한 액션 플랜을 제시하라.

[출력 형식]
## ⚡ 오늘의 액션 플랜

### 즉시 실행 (오늘)
각 종목마다 아래 중 하나를 명확히 선택하고 이유를 1~2줄로:
- 🟢 **비중 확대** — 구체적 이유
- 🔵 **홀드** — 구체적 이유
- 🔴 **비중 축소** — 구체적 이유
- ⚫ **전량 매도 고려** — 구체적 이유

### 이번 주 모니터링 포인트
반드시 체크해야 할 이벤트/지표 3가지

### 시나리오별 대응
**강세 시나리오**: (조건) → (행동)
**약세 시나리오**: (조건) → (행동)

[규칙]
- 모든 종목에 "홀드"를 추천하는 것은 금지
- "~할 수 있습니다", "~고려해볼 만합니다" 같은 애매한 표현 금지
- 과감하고 명확하게. 틀릴 수 있어도 방향성을 제시하라.
- 한국어로 작성

[시그널 분석]
{state['signal'][:1000]}

[포트폴리오 분석]
{state['portfolio_analysis'][:800]}
"""
    try:
        action_plan = llm.invoke(prompt).content
    except Exception as e:
        action_plan = f"(액션 플랜 생성 실패: {e})"
    return {"action_plan": action_plan}


def build_report_node(state: BriefingState) -> dict:
    idx_text = rows_to_text(state["indices"])
    port_text = rows_to_text(state["prices"], with_ticker=True)
    report = f"""# 📈 오늘의 투자 브리핑 ({state['today']})

## 1. 주요 시장 지수
{idx_text}

## 2. 내 보유 종목 현황
{port_text}

## 3. 뉴스 — 전체 시황
{state['market_news']}

## 4. 뉴스 — 내 보유 종목
{state['stock_news']}

## 5. 📡 Signal Agent
{state['signal']}

## 6. 📊 Portfolio Agent
{state['portfolio_analysis']}

## 7. ⚡ Advisor Agent
{state['action_plan']}
"""
    return {"report": report}


# ==========================================
# 4) Graph 조립
# ==========================================
def build_graph():
    g = StateGraph(BriefingState)
    g.add_node("collect_indices", collect_indices_node)
    g.add_node("collect_portfolio", collect_portfolio_node)
    g.add_node("collect_market_news", collect_market_news_node)
    g.add_node("collect_stock_news", collect_stock_news_node)
    g.add_node("signal_agent", signal_agent_node)
    g.add_node("portfolio_agent", portfolio_agent_node)
    g.add_node("advisor_agent", advisor_agent_node)
    g.add_node("build_report", build_report_node)

    g.add_edge(START, "collect_indices")
    g.add_edge("collect_indices", "collect_portfolio")
    g.add_edge("collect_portfolio", "collect_market_news")
    g.add_edge("collect_market_news", "collect_stock_news")
    g.add_edge("collect_stock_news", "signal_agent")
    g.add_edge("signal_agent", "portfolio_agent")
    g.add_edge("portfolio_agent", "advisor_agent")
    g.add_edge("advisor_agent", "build_report")
    g.add_edge("build_report", END)
    return g.compile()


def run_briefing(portfolio: List[dict]) -> BriefingState:
    app = build_graph()
    init_state = {
        "portfolio": portfolio,
        "today": datetime.now().strftime("%Y년 %m월 %d일"),
        # 아래 키들 초기화 추가
        "indices": [],
        "prices": [],
        "market_news": "",
        "stock_news": "",
        "signal": "",
        "portfolio_analysis": "",
        "action_plan": "",
        "report": "",
    }
    return app.invoke(init_state)


if __name__ == "__main__":
    pf = [
        {"name": "SK하이닉스", "ticker": "000660.KS"},
        {"name": "삼성전자", "ticker": "005930.KS"},
        {"name": "마이크론", "ticker": "MU"},
    ]
    result = run_briefing(pf)
    print(result["report"])
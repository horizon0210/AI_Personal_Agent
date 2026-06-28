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

# .env(로컬) 또는 Streamlit Secrets(배포) 양쪽에서 API 키 지원
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

INDICES = {"S&P 500": "^GSPC", "나스닥": "^IXIC", "코스피": "^KS11"}


# ==========================================
# 1) State 정의 — 노드 사이를 흐르는 데이터
# ==========================================
class BriefingState(TypedDict):
    portfolio: List[dict]
    today: str
    indices: List[dict]
    prices: List[dict]
    market_news: str
    stock_news: str
    advice: str          # 종목별 통합 조언
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
# 2) Node 정의
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
        results = safe_news(f"{item['name']} 주가", 2)
        if results:
            for r in results:
                out += f"🔹 **[{r['title']}]({r.get('url','')})**\n\n   {r.get('body','')}\n\n"
        else:
            out += "(뉴스를 불러오지 못했습니다)\n\n"
        time.sleep(1)
    return {"stock_news": out}


def generate_advice_node(state: BriefingState) -> dict:
    idx_text = rows_to_text(state["indices"])
    port_text = rows_to_text(state["prices"], with_ticker=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    prompt = f"""너는 전문 주식 투자 어드바이저다.
아래는 오늘 '실제로 수집된' 데이터다. 이 데이터만 근거로 조언을 작성하라.

[작성 형식]
- 보유한 '각 종목마다' 소제목(### 종목명)을 달고 2~3문장으로 조언하라.
- 각 종목 조언에는 반드시 다음을 포함하라:
  (1) 오늘 등락률과 관련 뉴스에 대한 해석
  (2) 해당 산업군 관점에서 '포트폴리오 비중'을 늘릴지/유지할지/줄일지에 대한 의견과 그 이유
- 마지막에 '### 종합' 소제목으로 포트폴리오 전체의 산업 편중/분산 관점 코멘트를 2~3문장 덧붙여라.

[규칙]
- 숫자나 사실을 새로 지어내지 말 것
- 단정적 매수/매도 단언 금지, 비중 조절은 '관점/이유'로 제시하고 리스크를 함께 언급
- 한국어로 간결하게

[주요 지수]
{idx_text}

[내 보유 종목]
{port_text}

[전체 시황 뉴스]
{state['market_news'][:1500]}

[보유 종목 뉴스]
{state['stock_news'][:1500]}
"""
    try:
        advice = llm.invoke(prompt).content
    except Exception as e:
        advice = f"(조언 생성 실패: {e})"
    return {"advice": advice}


def build_report_node(state: BriefingState) -> dict:
    idx_text = rows_to_text(state["indices"])
    port_text = rows_to_text(state["prices"], with_ticker=True)
    report = f"""# 📈 오늘의 투자 브리핑 ({state['today']})

## 1. 주요 시장 지수
{idx_text}

## 2. 내 보유 종목 현황
{port_text}

## 3. 뉴스 브리핑 — 전체 시황
{state['market_news']}

## 4. 뉴스 브리핑 — 내 보유 종목
{state['stock_news']}

## 5. 💡 종목별 투자 조언
{state['advice']}
"""
    return {"report": report}


# ==========================================
# 3) Graph 조립
# ==========================================
def build_graph():
    g = StateGraph(BriefingState)
    g.add_node("collect_indices", collect_indices_node)
    g.add_node("collect_portfolio", collect_portfolio_node)
    g.add_node("collect_market_news", collect_market_news_node)
    g.add_node("collect_stock_news", collect_stock_news_node)
    g.add_node("generate_advice", generate_advice_node)
    g.add_node("build_report", build_report_node)

    g.add_edge(START, "collect_indices")
    g.add_edge("collect_indices", "collect_portfolio")
    g.add_edge("collect_portfolio", "collect_market_news")
    g.add_edge("collect_market_news", "collect_stock_news")
    g.add_edge("collect_stock_news", "generate_advice")
    g.add_edge("generate_advice", "build_report")
    g.add_edge("build_report", END)
    return g.compile()


def run_briefing(portfolio: List[dict]) -> BriefingState:
    app = build_graph()
    init_state = {
        "portfolio": portfolio,
        "today": datetime.now().strftime("%Y년 %m월 %d일"),
    }
    return app.invoke(init_state)


# 단독 실행 테스트
if __name__ == "__main__":
    pf = [
        {"name": "SK하이닉스", "ticker": "000660.KS"},
        {"name": "삼성전자", "ticker": "005930.KS"},
        {"name": "마이크론", "ticker": "MU"},
    ]
    result = run_briefing(pf)
    print(result["report"])

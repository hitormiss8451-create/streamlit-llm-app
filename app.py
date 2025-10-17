from dotenv import load_dotenv

load_dotenv()
# app.py
# -*- coding: utf-8 -*-

import os
import streamlit as st

# --- 環境変数の読み込み（ローカルでは .env、Cloud では st.secrets が使われます）---
try:
    # python-dotenv が無い環境でも落ちないように try/except
    from dotenv import load_dotenv
    load_dotenv()  # .env があれば読み込む（ローカル開発用）
except Exception:
    pass

# --- LangChain のインポート（新しい推奨パス）---
# 旧: from langchain.chat_models import ChatOpenAI ではなく、
# 新: langchain_openai の ChatOpenAI を使います
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ========== ユーティリティ ==========

def get_api_key() -> str | None:
    """Streamlit Secrets または 環境変数 から OPENAI_API_KEY を取得"""
    # Streamlit Cloud では「App settings → Secrets」で設定した値が st.secrets から取れます
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    return key

SYSTEM_PROMPTS = {
    "A": "あなたは料理の専門家です。ユーザーの条件に具体的で再現性の高いレシピと調理のアドバイスをしてください。",
    "B": "あなたは経済/投資の専門家です。ユーザーの質問に対して具体的で実務的なアドバイスをしてください。"
}

def ask_expert(input_text: str, expert_choice: str, temperature: float = 0.2) -> str:
    if expert_choice not in SYSTEM_PROMPTS:
        raise ValueError("expert_choice は 'A' または 'B' を指定してください。")

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY が見つかりません。ローカルでは .env に "
            "OPENAI_API_KEY=... を記載、Streamlit Cloud では Secrets に設定してください。"
        )

    # LangChain OpenAI クライアント（最新の推奨経路）
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",  # コストと応答品質のバランスが良い。必要に応じて変更可。
        temperature=temperature,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPTS[expert_choice]),
        HumanMessage(content=input_text),
    ]

    result = llm.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)

# ========== Streamlit UI ==========

st.set_page_config(page_title="LLM Expert Helper", page_icon="🤖", layout="centered")

st.title("LLM Expert Helper 🤖")
st.caption("Python 3.11 / LangChain（langchain_openai）対応版・Streamlit Cloud 互換")

with st.form("main_form"):
    expert_choice = st.selectbox("専門家を選択", options=[("A", "料理の専門家"), ("B", "経済/投資の専門家")], index=0, format_func=lambda x: x[1])[0]
    user_input = st.text_area("質問内容を入力", placeholder="例: 鶏むね肉で作れる簡単で高タンパクなレシピは？", height=150)
    temperature = st.slider("出力の多様性（0=堅実, 1=多様）", 0.0, 1.0, 0.2, 0.05)
    submitted = st.form_submit_button("回答する")

if submitted:
    try:
        if not user_input.strip():
            st.warning("質問内容を入力してください。")
        else:
            with st.spinner("考えています..."):
                answer = ask_expert(user_input, expert_choice, temperature)
            st.success("回答")
            st.write(answer)
    except RuntimeError as e:
        # APIキー未設定などの致命的エラーはユーザーに分かりやすく表示
        st.error(str(e))
        st.info(
            "ローカル開発: プロジェクト直下に `.env` を作成して `OPENAI_API_KEY=...` を入れてください。\n"
            "Streamlit Cloud: App settings → Secrets に `OPENAI_API_KEY` を設定してください。"
        )
    except Exception as e:
        # それ以外は詳細を見せつつ落ちないように
        st.error(f"エラーが発生しました: {type(e).__name__}: {e}")

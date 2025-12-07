import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
import re

import openai
from openai import OpenAI

# =========================
# 0. OpenAI 설정
# =========================
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
MODEL_NAME = "gpt-4o-mini"


# =========================
# 1. DB 함수들
# =========================
def get_connection():
    conn = sqlite3.connect("fitness.db", check_same_thread=False)
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # 사용자 프로필 (username + password + 기본 정보)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            location TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # 운동 로그
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            log_date TEXT NOT NULL,
            exercise TEXT NOT NULL,
            amount INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def upsert_profile(username: str, password: str, age=None, sex=None, location=None):
    """username이 없으면 새로 만들고, 있으면 일부 정보만 업데이트"""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, password, age, sex, location FROM user_profiles WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()

    now = datetime.now().isoformat()

    if row is None:
        # 새 계정
        cur.execute(
            """
            INSERT INTO user_profiles (username, password, age, sex, location, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, password, age, sex, location, now, now),
        )
    else:
        # 기존 계정 → 패스워드는 그대로 두고, age/sex/location만 있을 때만 업데이트
        _, saved_pw, saved_age, saved_sex, saved_loc = row
        if saved_pw != password:
            # 비밀번호가 다르면 업데이트하지 않음
            conn.commit()
            conn.close()
            raise ValueError("비밀번호가 일치하지 않습니다.")

        new_age = saved_age if saved_age is not None else age
        new_sex = saved_sex if saved_sex is not None else sex
        new_loc = saved_loc if saved_loc is not None else location

        cur.execute(
            """
            UPDATE user_profiles
            SET age = ?, sex = ?, location = ?, updated_at = ?
            WHERE username = ?
            """,
            (new_age, new_sex, new_loc, now, username),
        )

    conn.commit()
    conn.close()


def get_profile(username: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT age, sex, location
        FROM user_profiles
        WHERE username = ?
        """,
        (username,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return {"age": None, "sex": None, "location": None}

    age, sex, loc = row
    return {"age": age, "sex": sex, "location": loc}


def insert_log(username, log_date, exercise, amount):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO logs (username, log_date, exercise, amount, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, log_date, exercise, amount, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_logs(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT log_date, exercise, amount, created_at
        FROM logs
        WHERE username = ?
        ORDER BY log_date DESC, created_at DESC
        """,
        (username,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_recent_stats(username, days: int = 30):
    """최근 N일 운동 요약 (일수, 가장 많이 한 운동, 총 운동량)"""
    rows = get_logs(username)
    if not rows:
        return {
            "days": 0,
            "top_exercise": None,
            "top_amount": 0,
            "total_amount": 0,
        }

    df = pd.DataFrame(rows, columns=["log_date", "exercise", "amount", "created_at"])
    # 문자열 날짜 → date
    df["log_date"] = pd.to_datetime(df["log_date"]).dt.date

    cutoff = date.today() - timedelta(days=days)
    df_recent = df[df["log_date"] >= cutoff]

    if df_recent.empty:
        return {
            "days": 0,
            "top_exercise": None,
            "top_amount": 0,
            "total_amount": 0,
        }

    days_count = df_recent["log_date"].nunique()
    total_amount = int(df_recent["amount"].sum())

    ex_group = df_recent.groupby("exercise")["amount"].sum().reset_index()
    ex_group = ex_group.sort_values("amount", ascending=False)
    top_row = ex_group.iloc[0]
    top_exercise = top_row["exercise"]
    top_amount = int(top_row["amount"])

    return {
        "days": int(days_count),
        "top_exercise": top_exercise,
        "top_amount": top_amount,
        "total_amount": total_amount,
    }


# =========================
# 2. 공공데이터(체력 기준표) 로드
# =========================
@st.cache_data
def load_norm_table():
    try:
        df = pd.read_csv("norm_table_202505_all_filtered.csv")
        return df
    except Exception:
        return None


norm_df = load_norm_table()


def simple_norm_comment(age: int, sex: str, exercise_name: str, value: float) -> str:
    if norm_df is None:
        return ""

    # 대충 나이대 → 그룹 매핑
    if age < 13:
        age_group = "유소년"
    elif age < 20:
        age_group = "청소년"
    elif age < 65:
        age_group = "성인"
    else:
        age_group = "어르신"

    metric_map = {
        "윗몸일으키기": "윗몸말아올리기(회)",
        "윗몸": "윗몸말아올리기(회)",
        "제자리 멀리뛰기": "제자리 멀리뛰기(cm)",
        "멀리뛰기": "제자리 멀리뛰기(cm)",
        "왕복오래달리기": "왕복오래달리기(회)",
    }

    target_metric = None
    for key, m in metric_map.items():
        if key in exercise_name:
            target_metric = m
            break

    if target_metric is None:
        return ""

    sub = norm_df[
        (norm_df["AGRDE_FLAG_NM"] == age_group)
        & (norm_df["sex"] == sex)
        & (norm_df["metric"] == target_metric)
    ]

    if sub.empty:
        return ""

    row = sub.iloc[0]
    mean = row["mean"]
    p30 = row["p30"]
    p70 = row["p70"]

    if value < p30:
        level = "하 (하위 30% 이하)"
    elif value > p70:
        level = "상 (상위 30% 수준)"
    else:
        level = "중 (중간 수준)"

    comment = (
        f"- 기준: {age_group} {sex}의 '{target_metric}' 평균은 약 {mean:.1f}, "
        f"30% 지점 {p30:.1f}, 70% 지점 {p70:.1f}야.\n"
        f"- 네 기록 {value:.1f} → **{level}** 정도로 볼 수 있어.\n"
    )
    return comment


# =========================
# 3. Rate Limit 시 fallback 멘트
# =========================
def simple_fallback_reply(user_input: str) -> str:
    base = (
        "지금은 AI 서버 쿼터 문제 때문에 고급 분석은 잠깐 막혀 있어 😢\n"
        "그래도 코치 입장에서 최대한 정리해서 말해볼게.\n\n"
    )

    text = user_input.lower()

    if "못했" in text or "안 했" in text or "안했" in text or "운동 안" in text:
        return (
            base
            + "오늘은 많이 못 움직였어도 괜찮아. 그런 날도 있는 거지 뭐 😊\n"
            + "지금 자리에서 스쿼트 10개, 팔굽 5개만 해볼까?\n"
            + "그리고 끝나고 **위쪽 탭에서 '오늘 운동 기록' 눌러서 방금 한 운동 기록**도 남겨줘! 내일 볼 때 훨씬 좋거든 🔥"
        )

    if "윗몸" in text:
        return (
            base
            + "복근 운동은 코어랑 자세 교정에 진짜 중요해.\n"
            + "주 3~4회, 3세트 x 15회 정도 해보자. 세트 사이에는 1분 정도 쉬고!\n"
            + "운동 끝나면 **'오늘 운동 기록' 탭에 오늘 윗몸일으키기 몇 개 했는지 꼭 적어줘** 😄"
        )

    if "달리기" in text or "조깅" in text or "뛰" in text:
        return (
            base
            + "달리기는 심폐지구력 올리는 데 최고야.\n"
            + "처음엔 '1분 뛰고 2분 걷기' 이런 식으로 15분만 채우는 걸 목표로 해보자.\n"
            + "다 하고 나서는 **'오늘 운동 기록' 탭에 오늘 뛴 시간이나 느낌** 한 줄 남겨줘. 꾸준함이 제일 중요해 🏃‍♂️"
        )

    return (
        base
        + "지금 상태랑 고민 말해준 것만으로도 이미 좋은 출발이야.\n"
        + "가볍게 스트레칭하고, 스쿼트 10개 + 팔벌려뛰기 20개 정도만 해도 몸이 확 달라져.\n"
        + "그리고 끝나면 **'오늘 운동 기록' 탭에 오늘 뭐 했는지 적는 것** 잊지 말기! 🙌"
    )


# =========================
# 4. Streamlit 초기 세팅
# =========================
st.set_page_config(page_title="AI 체력 코치", page_icon="💪", layout="wide")
init_db()

st.title("💪 대화만으로 내 체력을 분석하고, 운동 루틴과 근처 시설까지 추천해주는 AI")

# =========================
# 5. 사이드바 로그인
# =========================
st.sidebar.title("🙂 로그인")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "profile" not in st.session_state:
    st.session_state.profile = {"age": None, "sex": None, "location": None}
if "messages" not in st.session_state:
    st.session_state.messages = []

input_username = st.sidebar.text_input("닉네임 (아이디)", value="test1")
input_password = st.sidebar.text_input("비밀번호", type="password")

if st.sidebar.button("로그인 / 회원가입"):
    if not input_username.strip() or not input_password.strip():
        st.sidebar.error("아이디와 비밀번호를 모두 입력해줘!")
    else:
        try:
            # 프로필 정보는 나중에 채워도 되니까 여기선 기본값만 저장/검증
            upsert_profile(input_username.strip(), input_password.strip())
            st.session_state.logged_in = True
            st.session_state.username = input_username.strip()
            st.session_state.profile = get_profile(st.session_state.username)
            st.sidebar.success(f"{st.session_state.username} 로그인 완료!")
        except ValueError as e:
            st.sidebar.error(str(e))

if not st.session_state.logged_in or not st.session_state.username:
    st.write("⬅️ 왼쪽 사이드바에서 아이디/비밀번호를 입력하고 **로그인 / 회원가입** 버튼을 눌러줘.")
    st.stop()

current_user = st.session_state.username
st.sidebar.markdown(f"**현재 사용자:** {current_user}")


# 로그인 후 최신 프로필 정보 반영
st.session_state.profile = get_profile(current_user)


# =========================
# 6. 탭 구성
# =========================
tab_chat, tab_log, tab_history, tab_summary = st.tabs(
    ["🧠 AI 체력 코치", "📝 오늘 운동 기록", "📚 기록 보기", "📊 요약 & 피드백"]
)


# -------------------------
# 6-1. AI 체력 코치 탭
# -------------------------
with tab_chat:
    st.subheader("🧠 AI 체력 코치")

    profile = st.session_state.profile
    stats = get_recent_stats(current_user)

    # 대화가 전혀 없으면 → 첫 인삿말 자동 추가 (프로필/기록 여부에 따라 다르게)
    if len(st.session_state.messages) == 0:
        if profile["age"] and profile["sex"] and profile["location"]:
            # 프로필 + 기록 기반 요약형 인사
            days = stats["days"]
            top_ex = stats["top_exercise"]
            top_amt = stats["top_amount"]
            total_amt = stats["total_amount"]

            if days == 0:
                summary_text = (
                    f"오! {current_user} 다시 왔구나 😄\n\n"
                    "아직 최근 30일 동안 저장된 운동 기록은 없어. 지금이 진짜 1일 차야!🔥\n"
                    f"프로필은 대충 이렇게 알고 있어:\n"
                    f"- 나이: {profile['age']}살\n"
                    f"- 성별: {profile['sex']}\n"
                    f"- 운동하는 동네: {profile['location']}\n\n"
                    "오늘 뭐부터 해볼지 같이 정해볼까?\n"
                    "운동 끝나면 **위에 `오늘 운동 기록` 탭 눌러서 오늘 한 운동도 기록해줘!**"
                )
            else:
                summary_text = (
                    f"오! {current_user} 다시 왔구나 😄\n\n"
                    f"최근 30일 기준으로 정리해보면,\n"
                    f"- 운동한 날: {days}일\n"
                    f"- 가장 많이 한 운동: {top_ex} (누적 {top_amt} 단위)\n"
                    f"- 총 운동량: {total_amt} 단위 정도야.\n\n"
                    f"프로필은 대충 이렇게 알고 있어:\n"
                    f"- 나이: {profile['age']}살\n"
                    f"- 성별: {profile['sex']}\n"
                    f"- 운동하는 동네: {profile['location']}\n\n"
                    "오늘 몸 상태가 어떤지, 그리고 어떤 운동을 해보고 싶은지 말해줘!\n"
                    "참, 운동 끝나면 **위에 `오늘 운동 기록` 탭에 오늘 한 운동 기록** 남겨주면 내가 보기 더 편해 😊"
                )
        else:
            # 아직 프로필이 완전치 않을 때 → 질문형 인사
            summary_text = (
                f"안녕 {current_user}! 나는 너 전용 AI 체력 코치야 💪\n\n"
                "너를 좀 알아야 제대로 도와줄 수 있어서, 몇 가지만 편하게 말해줘!\n\n"
                "- 나이는 몇 살이야?\n"
                "- 성별은? (남 / 여)\n"
                "- 달리기는 어느 정도야? (예: 10분만 뛰어도 숨차 / 30분은 가능 등)\n"
                "- 스쿼트는 한 번에 몇 개 정도 할 수 있어?\n"
                "- 보통 어느 동네에서 운동해? (예: 강남구 대치동)\n\n"
                "한 번에 길게 써도 되고, 하나씩 나눠서 말해도 돼 😄\n"
                "그리고 운동 끝나면 **`오늘 운동 기록` 탭에 오늘 한 운동도 꼭 기록해줘!**"
            )

        st.session_state.messages.append({"role": "assistant", "content": summary_text})

    # 이전 대화 모두 출력
    for msg in st.session_state.messages:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.markdown(msg["content"])

    # === 채팅 입력은 항상 맨 아래에 위치 ===
    user_input = st.chat_input("오늘 몸 상태나 목표, 고민을 편하게 말해줘!")

    if user_input:
        # 1) 유저 메시지 저장 + 출력
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) 입력에서 프로필 정보 추출 (나이/성별/동네)
        age_match = re.search(r"(\d+)\s*살", user_input)
        new_age = None
        new_sex = None
        new_loc = None

        if age_match:
            new_age = int(age_match.group(1))

        if any(x in user_input for x in ["남자", "남성", "남 "]):
            new_sex = "남"
        elif any(x in user_input for x in ["여자", "여성", "여 "]):
            new_sex = "여"

        if "구" in user_input or "동" in user_input or "시" in user_input:
            # 대충 문장 전체를 location 후보로 넣고, 프롬프트에서 정제하게 둠
            new_loc = user_input

        # 프로필에 새로운 정보 있으면 DB 업데이트
        if new_age is not None or new_sex is not None or new_loc is not None:
            try:
                upsert_profile(
                    current_user,
                    input_password if input_password else "",  # 이미 로그인한 상태라 실제로는 pw 안 씀
                    age=new_age,
                    sex=new_sex,
                    location=new_loc,
                )
            except Exception:
                pass  # 여기선 조용히 무시
            st.session_state.profile = get_profile(current_user)
            profile = st.session_state.profile

        # 3) 윗몸일으키기 등 간단 기준 비교
        extra_analysis = ""
        if profile["age"] and profile["sex"]:
            situp_match = re.search(r"(윗몸일으키기|윗몸)\D*(\d+)\s*개", user_input)
            if situp_match:
                sit_val = float(situp_match.group(2))
                extra_analysis = simple_norm_comment(
                    profile["age"], profile["sex"], "윗몸일으키기", sit_val
                )

        # 4) 시스템 프롬프트 구성 (반말 + Encourage + 기록탭 리마인드 포함)
        base_system_prompt = """
너는 '스포츠 공공데이터 기반 퍼스널 체력 분석 AI 코치'야.
항상 **반말**로, 친구같이 편하지만 **전문적인 트레이너** 느낌으로 말해.

역할:
- 사용자의 나이, 성별, 운동 수준(달리기, 스쿼트, 턱걸이 등), 거주 동네를 자연스럽게 질문하면서 알아가.
- 국민 체력측정/생활체육 통계 같은 걸 참고하는 코치인 것처럼,
  "대략 이 정도면 상/중/하" 식으로 구체적인 피드백을 준다.
- 매번 답변에서:
  1) 첫 문단에서 현재 체력 상태를 한 줄로 요약해줘.
  2) 그 다음에는 bullet 형식으로
     - 현재 체력 레벨 (상/중/하 느낌)
     - 오늘 할 핵심 운동 루틴 (세트 × 반복, 강도, 휴식 구체적으로)
     - 1주일 정도의 짧은 목표
     를 제시해.
  3) 마지막에는 짧게 1~2문장 정도로 동기부여 멘트를 넣어줘 (길게 감성 소설 쓰지 말 것).

- 사용자가 동네나 '마포구 대흥동', '강남구 대치동' 같은 표현을 말하면,
  그 주변에 있을 법한 운동 장소 유형(한강 러닝코스, 동네 공원, 헬스장, 체력인증센터 등)을
  구체적으로 예시로 들어줘.

- 아주 중요:
  답변 중간에 **가끔씩** (예: 2~3번 답변에 한 번 정도) 자연스럽게
  '운동 끝나면 위쪽 탭에 있는 **`오늘 운동 기록` 탭에 들어가서 오늘 한 운동 기록을 남겨달라'는
  리마인드 멘트를 섞어줘.
  하지만 매 답변마다 강요하진 말고, 자연스럽게 말투에 섞어서 이야기해.

말투 예시:
- "이 정도면 상체 힘은 꽤 괜찮은 편이야."
- "오늘 루틴은 이렇게 가보자."
- "운동 끝나면 오늘 한 거 잊기 전에 '오늘 운동 기록' 탭에 살짝 적어두면 나중에 내가 분석하기도 좋아!"
"""

        # 최근 기록/프로필 요약을 system에 같이 태움
        stats = get_recent_stats(current_user)
        stats_text = (
            f"최근 30일 기준 요약: 운동한 날 {stats['days']}일, "
            f"가장 많이 한 운동: {stats['top_exercise']}, "
            f"해당 누적량: {stats['top_amount']} 단위, "
            f"총 운동량: {stats['total_amount']} 단위.\n"
        )

        profile_text = (
            f"현재까지 파악된 프로필: 나이={profile['age']}, 성별={profile['sex']}, "
            f"운동 지역 관련 입력={profile['location']}.\n"
        )

        system_prompt = base_system_prompt + "\n\n" + stats_text + profile_text
        if extra_analysis:
            system_prompt += (
                "\n아래는 백엔드에서 계산한 대략적인 체력 기준 비교 결과야. "
                "이 내용을 참고해서 더 구체적으로 피드백해줘.\n"
            )
            system_prompt += extra_analysis + "\n"

        # 5) OpenAI 호출
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}]
                + st.session_state.messages,
                max_tokens=700,
                temperature=0.7,
            )
            bot_reply = response.choices[0].message.content

        except openai.RateLimitError:
            bot_reply = simple_fallback_reply(user_input)
            st.warning(
                "⚠️ 현재 OpenAI API 쿼터가 부족해서, "
                "고급 분석 대신 간단 코치 모드로 답변할게!"
            )
        except Exception as e:
            bot_reply = (
                "AI 코치 호출 중 오류가 났어 😢\n"
                f"에러 내용: {str(e)}\n\n"
                "그래도 운동에 대해 궁금한 거 있으면 편하게 물어봐줘. "
                "일반 코치 모드로라도 최대한 도와볼게!"
            )

        # 6) 답변 저장 + 출력
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)


# -------------------------
# 6-2. 오늘 운동 기록 탭
# -------------------------
with tab_log:
    st.subheader("📝 오늘 운동 기록 남기기")

    col1, col2 = st.columns(2)
    with col1:
        log_date = st.date_input("운동한 날짜", value=date.today())
    with col2:
        exercise = st.selectbox(
            "운동 종류",
            ["팔굽혀펴기", "윗몸일으키기", "스쿼트", "달리기(분)", "턱걸이", "플랭크(초)", "기타"],
        )

    amount = st.number_input(
        "운동 양 (횟수 / 시간)", min_value=1, max_value=10000, value=20, step=1
    )

    if st.button("기록 저장하기"):
        insert_log(
            username=current_user,
            log_date=log_date.isoformat(),
            exercise=exercise,
            amount=int(amount),
        )
        st.success("운동 기록 저장 완료! 🔥\n이제 챗 탭으로 돌아가면, 내가 이 기록도 반영해서 얘기해줄게.")


# -------------------------
# 6-3. 기록 보기 탭
# -------------------------
with tab_history:
    st.subheader("📚 내 운동 기록")

    rows = get_logs(current_user)
    if not rows:
        st.info("아직 기록이 없어. 운동하고 나서 **'오늘 운동 기록' 탭**에서 한 번 적어보자! 😄")
    else:
        df = pd.DataFrame(rows, columns=["날짜", "운동", "양", "기록 시간"])
        st.dataframe(df, use_container_width=True)


# -------------------------
# 6-4. 요약 & 피드백 탭
# -------------------------
with tab_summary:
    st.subheader("📊 최근 운동 요약 & 간단 피드백")

    rows = get_logs(current_user)
    if not rows:
        st.info("아직 분석할 운동 기록이 없어 😅\n오늘 뭔가 하나라도 하고 기록부터 남겨보자!")
    else:
        df = pd.DataFrame(rows, columns=["날짜", "운동", "양", "기록 시간"])

        summary = (
            df.groupby("날짜")["양"].sum().reset_index().sort_values("날짜")
        )

        st.write("📈 최근 운동량 (날짜별 합계)")
        st.line_chart(summary, x="날짜", y="양")

        total_days = summary["날짜"].nunique()
        total_amount = int(summary["양"].sum())

        st.markdown(f"- 최근 운동한 날 수: **{total_days}일**")
        st.markdown(f"- 총 운동량(단순 합 기준): **{total_amount} 단위**")

        if total_days == 0:
            msg = "이제 막 시작 단계야! 오늘 가볍게 5분만이라도 움직여볼까? 😊"
        elif total_days < 3:
            msg = "좋아, 시동이 걸리고 있어. 이번 주 3일만 채워보자 💪"
        elif total_days < 7:
            msg = "꾸준함이 보인다. 주 3~4일 운동이면 이미 상위권이야 🤫"
        else:
            msg = "와… 이 정도면 주변 사람들한테 건강 전도사 해도 될 수준이야 🔥 계속 가보자!"

        st.markdown("### 🧠 요약 코멘트")
        st.success(msg)

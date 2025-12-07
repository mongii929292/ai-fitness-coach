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

    # 운동 기록 테이블
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

    # 사용자 프로필 테이블
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            run_level TEXT,
            squat_level TEXT,
            location TEXT
        )
        """
    )

    conn.commit()
    conn.close()


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


def create_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, password),
    )
    conn.commit()
    conn.close()


def get_user(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, password, age, sex, run_level, squat_level, location
        FROM users
        WHERE username = ?
        """,
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def update_user_profile(username, profile: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE users
        SET age = ?, sex = ?, run_level = ?, squat_level = ?, location = ?
        WHERE username = ?
        """,
        (
            profile.get("age"),
            profile.get("sex"),
            profile.get("run_level"),
            profile.get("squat_level"),
            profile.get("location"),
            username,
        ),
    )
    conn.commit()
    conn.close()


# =========================
# 2. 공공데이터 로드 (옵션)
# =========================
@st.cache_data
def load_norm_table():
    try:
        df = pd.read_csv("norm_table_202505_all_filtered.csv")
        return df
    except Exception:
        return None


@st.cache_data
def load_facility_table():
    try:
        df = pd.read_csv("전국체육시설_전체데이터.csv")
        return df
    except Exception:
        return None


norm_df = load_norm_table()
facility_df = load_facility_table()


def simple_norm_comment(age: int, sex: str, exercise_name: str, value: float) -> str:
    if norm_df is None:
        return ""

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
        f"30% 지점 {p30:.1f}, 70% 지점 {p70:.1f}.\n"
        f"- 현재 기록 {value:.1f} → **{level}** 정도로 볼 수 있어.\n"
    )
    return comment


def extract_profile_from_text(text: str) -> dict:
    text = text.strip()
    result = {}

    # 나이
    age_match = re.search(r"나이(?:는)?\s*(\d+)", text)
    if not age_match:
        age_match = re.search(r"(\d+)\s*살", text)
    if age_match:
        try:
            result["age"] = int(age_match.group(1))
        except ValueError:
            pass

    # 성별
    if any(k in text for k in ["남자", "남성", " 남 "]):
        result["sex"] = "남"
    elif any(k in text for k in ["여자", "여성", " 여 "]):
        result["sex"] = "여"

    # 달리기 수준 문장 통째로 저장
    if "달리기" in text or "조깅" in text or "뛰" in text:
        result.setdefault("run_level", text)

    # 스쿼트 개수
    squat_match = re.search(r"스쿼트[^0-9]*(\d+)\s*(개|번)?", text)
    if squat_match:
        result["squat_level"] = squat_match.group(1)

    # 위치
    loc_match = re.search(r"([가-힣]+시\s*)?[가-힣]+구\s*[가-힣0-9]+동", text)
    if not loc_match:
        loc_match = re.search(r"[가-힣]+구", text)

    if loc_match:
        result["location"] = loc_match.group(0)

    return result


def build_facility_hint(location: str) -> str:
    if facility_df is None or not location:
        return ""

    try:
        df = facility_df.copy()
        cols = df.columns

        addr_cols = [c for c in cols if "addr" in c.lower() or "주소" in c]
        name_col = None
        for cand in ["faci_nm", "시설명", "FACI_NM"]:
            if cand in cols:
                name_col = cand
                break
        type_col = None
        for cand in ["ftype_nm", "fcob_nm", "시설유형"]:
            if cand in cols:
                type_col = cand
                break

        if name_col is None or not addr_cols:
            return ""

        mask = False
        for ac in addr_cols:
            mask = mask | df[ac].astype(str).str.contains(location, na=False)

        sub = df[mask].head(5)
        if sub.empty:
            return ""

        lines = []
        for _, row in sub.iterrows():
            nm = str(row[name_col])
            tp = str(row[type_col]) if type_col and pd.notna(row[type_col]) else ""
            addr = ""
            for ac in addr_cols:
                if pd.notna(row[ac]):
                    addr = str(row[ac])
                    break
            line = f"- 시설명: {nm}"
            if tp:
                line += f" / 유형: {tp}"
            if addr:
                line += f" / 주소: {addr}"
            lines.append(line)

        if not lines:
            return ""

        hint = (
            f"사용자가 말한 지역 '{location}' 기준으로 백엔드에서 추려본 체육시설 후보야:\n"
            + "\n".join(lines)
        )
        return hint
    except Exception:
        return ""


def simple_fallback_reply(user_input: str) -> str:
    base = (
        "지금은 AI 서버 쿼터 문제 때문에 고급 분석은 잠시 제한돼 있어.\n"
        "그래도 코치 입장에서 한 번 정리해볼게.\n\n"
    )

    text = user_input.lower()

    if "못했" in text or "안 했" in text or "안했" in text or "운동 안" in text:
        return (
            base
            + "오늘은 많이 못 움직인 날이네. 괜찮아, 누구나 그런 날 있어 😊\n"
            + "지금 자리에서 스쿼트 10개, 팔굽혀펴기 5개만 해볼까?\n"
            + "내일은 오늘보다 딱 1분만 더 움직이는 걸 목표로 잡자!"
        )

    if "윗몸" in text:
        return (
            base
            + "복근 운동은 코어 안정성과 자세 교정에 진짜 중요해.\n"
            + "주 3~4회, 세트 사이 1분 휴식 기준으로 3세트 정도를 추천해.\n"
            + "허리가 불편하면 상체를 너무 높이 들지 말고 통증 없는 범위에서만 해줘!"
        )

    if "달리기" in text or "조깅" in text or "뛰" in text:
        return (
            base
            + "달리기는 심폐지구력 올려주는 최고급 운동이야.\n"
            + "처음엔 '말하면서 숨 약간 찰 정도' 강도로 20분만 꾸준히 해봐.\n"
            + "일주일에 3번만 해도 2~4주 뒤 체력이 확 달라질 거야 🏃‍♂️"
        )

    return (
        base
        + "지금 상태랑 고민 말해준 것만으로도 이미 첫 걸음은 뗀 거야.\n"
        + "가벼운 스트레칭, 스쿼트 10개, 팔 벌려뛰기 20개부터 시작해 보자.\n"
        + "작은 습관이 쌓이면 체력은 생각보다 금방 좋아져 🙌"
    )


def is_profile_complete(profile: dict) -> bool:
    return all(
        profile.get(k) not in [None, "", 0]
        for k in ["age", "sex", "run_level", "squat_level", "location"]
    )


def get_user_summary(username: str):
    rows = get_logs(username)
    if not rows:
        return {
            "total_days_30": 0,
            "total_amount_30": 0,
            "top_exercise": None,
            "records_df": None,
        }

    df = pd.DataFrame(rows, columns=["log_date", "exercise", "amount", "created_at"])
    df["log_date"] = pd.to_datetime(df["log_date"])
    today = pd.to_datetime(date.today())
    since = today - timedelta(days=30)
    df_30 = df[df["log_date"] >= since]

    if df_30.empty:
        return {
            "total_days_30": 0,
            "total_amount_30": 0,
            "top_exercise": None,
            "records_df": df,
        }

    days = df_30["log_date"].dt.date.nunique()
    total_amount = df_30["amount"].sum()
    top_ex = (
        df_30.groupby("exercise")["amount"].sum().sort_values(ascending=False).index[0]
    )

    return {
        "total_days_30": int(days),
        "total_amount_30": int(total_amount),
        "top_exercise": top_ex,
        "records_df": df,
    }


# =========================
# 3. Streamlit 기본 세팅
# =========================
st.set_page_config(page_title="AI 체력 코치", page_icon="💪", layout="wide")
init_db()

st.title("💪 대화만으로 내 체력을 분석하고, 운동 루틴과 근처 시설까지 추천해주는 AI 서비스")


# =========================
# 4. 로그인 / 회원가입
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "profile" not in st.session_state:
    st.session_state.profile = {
        "age": None,
        "sex": None,
        "run_level": None,
        "squat_level": None,
        "location": None,
    }
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None

st.sidebar.header("🔐 로그인")

login_mode = st.sidebar.radio("모드 선택", ["로그인", "회원가입"], horizontal=True)
input_username = st.sidebar.text_input("닉네임(아이디)")
input_password = st.sidebar.text_input("비밀번호", type="password")

if login_mode == "회원가입":
    if st.sidebar.button("회원가입"):
        if not input_username or not input_password:
            st.sidebar.error("닉네임과 비밀번호를 모두 입력해줘!")
        else:
            existing = get_user(input_username)
            if existing:
                st.sidebar.error("이미 존재하는 닉네임이야. 다른 이름 써줘!")
            else:
                create_user(input_username, input_password)
                st.sidebar.success("회원가입 완료! 이제 '로그인' 탭에서 로그인 해줘.")

elif login_mode == "로그인":
    if st.sidebar.button("로그인"):
        if not input_username or not input_password:
            st.sidebar.error("닉네임과 비밀번호를 모두 입력해줘!")
        else:
            user_row = get_user(input_username)
            if not user_row:
                st.sidebar.error("해당 닉네임의 계정이 없어. 먼저 회원가입해줘!")
            else:
                _, db_pw, age, sex, run_level, squat_level, location = user_row
                if db_pw != input_password:
                    st.sidebar.error("비밀번호가 틀렸어 😅")
                else:
                    st.sidebar.success("로그인 성공!")
                    st.session_state.logged_in = True
                    st.session_state.username = input_username
                    st.session_state.profile = {
                        "age": age,
                        "sex": sex,
                        "run_level": run_level,
                        "squat_level": squat_level,
                        "location": location,
                    }
                    st.session_state.messages = []
                    st.session_state.greeted = False
                    st.session_state.pending_user_input = None

if not st.session_state.logged_in or not st.session_state.username:
    st.info("왼쪽에서 로그인해야 사용할 수 있어!")
    st.stop()

current_user = st.session_state.username
profile = st.session_state.profile

st.sidebar.success(f"현재 로그인: {current_user}")


# =========================
# 5. 탭 구성
# =========================
tab_chat, tab_log, tab_history, tab_summary = st.tabs(
    ["🧠 AI 코치와 대화", "📝 오늘 운동 기록", "📚 기록 보기", "📊 요약 & 피드백"]
)


# -------------------------
# 5-1. AI 코치와 대화 탭
# -------------------------
with tab_chat:
    # 수정: "(반말 모드)" 제거
    st.subheader("🧠 AI 체력 코치")

    # 1) 첫 인사 메시지 (로그/프로필 기반 요약) – 딱 한 번
    if not st.session_state.greeted:
        summary = get_user_summary(current_user)
        days_30 = summary["total_days_30"]
        total_amt_30 = summary["total_amount_30"]
        top_ex = summary["top_exercise"]

        prof_txt = []
        if profile.get("age"):
            prof_txt.append(f"{profile['age']}살")
        if profile.get("sex"):
            prof_txt.append(profile["sex"])
        if profile.get("location"):
            prof_txt.append(profile["location"])

        prof_str = " / ".join([p for p in prof_txt if p])

        if days_30 == 0:
            workout_line = "최근 30일 동안 기록된 운동이 아직 없어. 오늘이 진짜 1일 차야!🔥"
        else:
            workout_line = (
                f"최근 30일 동안 {days_30}일 운동했고, "
                f"가장 많이 한 운동은 **{top_ex}**, 총 운동량은 {total_amt_30} 단위 정도야."
            )

        if prof_str:
            header_line = (
                f"오! {current_user} 다시 왔네 😄\n\n"
                f"지금까지 내가 알고 있는 너 정보는 대략 이렇게야:\n"
                f"- {prof_str}\n"
                f"- {workout_line}\n\n"
                "오늘은 어떤 느낌이야? 몸 상태나 목표 편하게 말해줘!"
            )
        else:
            header_line = (
                f"오! {current_user} 환영해 😄\n\n"
                f"{workout_line}\n\n"
                "너에 대해 조금 더 알려주면 루틴이랑 장소까지 제대로 짜줄 수 있어.\n"
                "예시: '24살 남자, 달리기는 10분만 뛰어도 숨차고, 스쿼트는 20개 정도, 마포구 대흥동' 이런 식으로!"
            )

        st.session_state.messages.append({"role": "assistant", "content": header_line})
        st.session_state.greeted = True

    # 2) 대기 중인 입력(pending_user_input)이 있으면, 지금 턴에서 처리
    pending = st.session_state.pending_user_input
    if pending:
        user_text = pending

        # (a) 유저 메시지를 history에 추가
        st.session_state.messages.append({"role": "user", "content": user_text})

        # (b) 프로필 업데이트
        new_info = extract_profile_from_text(user_text)
        updated_profile = st.session_state.profile.copy()
        changed = False
        for k, v in new_info.items():
            if v and updated_profile.get(k) != v:
                updated_profile[k] = v
                changed = True

        st.session_state.profile = updated_profile
        profile = updated_profile

        if changed:
            update_user_profile(current_user, profile)

        # (c) 체력 기준 분석
        extra_analysis = ""
        if profile.get("age") and profile.get("sex"):
            situp_match = re.search(r"(윗몸일으키기|윗몸)\D*(\d+)\s*개", user_text)
            if situp_match:
                situp_value = float(situp_match.group(2))
                extra_analysis += simple_norm_comment(
                    profile["age"], profile["sex"], "윗몸일으키기", situp_value
                )

        # (d) 시설 힌트
        facility_hint = ""
        if profile.get("location"):
            facility_hint = build_facility_hint(profile["location"])

        # (e) 시스템 프롬프트 구성
        base_system_prompt = """
너는 '스포츠 과학 전공 + 퍼스널 트레이너 감성'을 가진 AI 체력 코치다.
항상 반말을 쓰고, 무조건 사용자를 칭찬하고 격려해 줘.

대답 규칙:

1) 첫 문장:
   - 지금까지 들은 정보 기준으로 사용자의 상태를 한 줄로 요약 + 칭찬 한 번.
     예: "24살 남자인데, 상체 힘은 꽤 괜찮고 유산소만 조금 더 키우면 좋겠어. 이미 잘하고 있어!"

2) 정보 부족 여부에 따라:

   (A) 아직 나이, 성별, 달리기 수준, 스쿼트 수준, 운동 지역 중 모르는 게 있으면
       → 오늘 루틴을 길게 짜지 말고 '질문 위주'로 대답한다.
         - 이때도 한두 줄 정도는 간단한 조언/응원은 해도 된다.
         - "이것만 더 알면 루틴이랑 장소까지 진짜 제대로 짜줄 수 있어" 같은 식으로 유도.

   (B) 나이, 성별, 달리기 수준, 스쿼트 수준, 운동 지역 정보가 다 채워져 있으면
       → 그때부터는 더 이상 정보만 달라고 하지 말고, '항상' 아래 구조를 지킨다:

       (1) '오늘 추천 운동 루틴' 섹션
           - 세트 x 반복, 강도(가볍게/중간/빡세게), 세트 간 휴식까지 구체적으로.
           - 상체/하체/코어/유산소 중에서 오늘 포커스를 1~2개 정해서 말해준다.

       (2) '1주 또는 4주 목표' 섹션
           - 너무 거창하지 않은 작은 목표 한 줄 (예: "이번 주에 달리기 총 40분 채우기").

       (3) '오늘 추천 운동 장소' 섹션  (무조건 포함)
           - 사용자의 지역 정보를 활용해서,
             예: "마포구 대흥동 기준으로"
             - 근처 공원 이름, 운동장, 체육공원, 헬스장, 체력인증센터 등
             실제 이름을 1~2개 콕 집어서 추천한다.
           - 먼저 핵심 추천 1곳을 말하고, 그 다음에 1~2개 정도 대안 장소를 짧게 덧붙인다.

       (4) 루틴과 장소 추천 후, 꼭 다음 두 가지를 모두 포함해야 한다:
           - **첫째:** 운동이 끝나면 **'📝 오늘 운동 기록' 탭**에 저장해달라고 요청한다.
           - **둘째:** '짧은 질문 딱 하나'만 던져서 대화를 이어간다.
             예: "오늘은 이 루틴으로 가볼까?", "실내/실외 중에 뭐가 더 끌려?" 등.
             (주의: 루틴과 장소를 먼저 충분히 제안한 뒤에 질문해야 함)

3) 특히 사용자가 '어디서 운동할까', '어디가 좋을까'처럼 장소를 물어보는 경우에는
   - 다시 질문으로 되묻지 말고, 먼저 답을 내린다.
     예: "마포구 대흥동이면 오늘은 **○○공원**에서 조깅 + 맨몸운동 세트로 해보자."
   - 그 다음에 "이렇게 해볼래?" 정도로만 가볍게 물어본다.

4) 말투 스타일:
   - 친구처럼 반말이지만, 설명은 꽤 구체적으로 (전문성 있는 느낌).
   - 너무 장문 소설처럼 쓰지 말고, 핵심만 쫀득하게.
"""

        system_prompt = base_system_prompt + "\n\n"
        system_prompt += "현재까지 파악된 사용자 프로필:\n"
        system_prompt += f"- 나이: {profile.get('age')}\n"
        system_prompt += f"- 성별: {profile.get('sex')}\n"
        system_prompt += f"- 달리기 수준 관련 문장: {profile.get('run_level')}\n"
        system_prompt += f"- 스쿼트 수준: {profile.get('squat_level')}\n"
        system_prompt += f"- 주 운동 지역: {profile.get('location')}\n"
        system_prompt += (
            f"- 프로필 완성도: {'완료' if is_profile_complete(profile) else '미완료'}\n"
        )

        if extra_analysis:
            system_prompt += "\n[백엔드 체력 기준 비교 예시]\n" + extra_analysis + "\n"

        if facility_hint:
            system_prompt += (
                "\n[백엔드에서 찾은 체육시설 후보 리스트]\n"
                + facility_hint
                + "\n이 후보들을 참고해서 실제 답변에서 1~2개만 골라 구체적으로 언급해줘.\n"
            )

        # OpenAI 호출
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages,
                ],
                max_tokens=700,
                temperature=0.7,
            )
            bot_reply = response.choices[0].message.content
        except openai.RateLimitError:
            bot_reply = simple_fallback_reply(user_text)
            st.warning(
                "⚠️ 현재 OpenAI API 쿼터가 부족해서, "
                "고급 분석 대신 간단한 코치 모드로 답변할게."
            )
        except Exception as e:
            bot_reply = (
                "AI 코치 호출 중 오류가 발생했어 😢\n"
                f"에러 내용: {str(e)}\n\n"
                "그래도 운동 관련해서 궁금한 점을 적어주면, "
                "일반 코치 모드로 최대한 도와볼게!"
            )

        # assistant 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        # 처리 끝났으니 pending 비우기
        st.session_state.pending_user_input = None

    # 3) 지금까지 메시지 전부 렌더링 (항상 입력창 위에만 나오도록)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 4) 입력창은 항상 맨 마지막에
    new_input = st.chat_input("여기에 그냥 편하게 써줘 😄")
    if new_input:
        st.session_state.pending_user_input = new_input
        st.rerun()


# -------------------------
# 5-2. 오늘 운동 기록 탭
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
        "운동 양 (횟수 / 시간 / 초)", min_value=1, max_value=10000, value=20, step=1
    )

    if st.button("기록 저장하기"):
        insert_log(
            username=current_user,
            log_date=log_date.isoformat(),
            exercise=exercise,
            amount=int(amount),
        )
        st.success("운동 기록이 저장됐어! 🔥")


# -------------------------
# 5-3. 기록 보기 탭
# -------------------------
with tab_history:
    st.subheader("📚 내 운동 기록")

    rows = get_logs(current_user)
    if not rows:
        st.info("아직 기록이 없어. 오늘 첫 운동을 기록해보자! 😄")
    else:
        df = pd.DataFrame(rows, columns=["log_date", "exercise", "amount", "created_at"])
        df_display = df.rename(
            columns={
                "log_date": "날짜",
                "exercise": "운동",
                "amount": "양",
                "created_at": "기록 시간",
            }
        )
        st.dataframe(df_display, use_container_width=True)


# -------------------------
# 5-4. 요약 & 피드백 탭
# -------------------------
with tab_summary:
    st.subheader("📊 최근 운동 요약 & 간단 피드백")

    rows = get_logs(current_user)
    if not rows:
        st.info("아직 기록이 없어서 분석할 데이터가 없어 😅 오늘부터 한 줄씩 쌓아보자!")
    else:
        df = pd.DataFrame(rows, columns=["log_date", "exercise", "amount", "created_at"])
        df["log_date"] = pd.to_datetime(df["log_date"])

        df_group = df.groupby("log_date")["amount"].sum().reset_index()
        df_group = df_group.sort_values("log_date")

        df_group_display = df_group.rename(columns={"log_date": "날짜", "amount": "총 운동량"})

        st.write("📈 최근 운동량 (날짜별 합계)")
        st.line_chart(df_group_display, x="날짜", y="총 운동량")

        total_days = df_group_display["날짜"].dt.date.nunique()
        total_amount = int(df_group_display["총 운동량"].sum())

        st.markdown(f"- 운동한 날 수: **{total_days}일**")
        st.markdown(f"- 총 운동량(단순 합): **{total_amount} 단위**")

        if total_days == 0:
            msg = "이제 막 시작 단계야! 오늘 한 번만이라도 가볍게 움직여보자 😊"
        elif total_days < 3:
            msg = "좋아, 시동이 걸리고 있어. 이번 주 3일만 채워보자! 💪"
        elif total_days < 7:
            msg = "꾸준함이 보인다. 주 3~4일 운동이면 이미 상위권이야 🤫"
        else:
            msg = "와… 이 정도면 주변 사람들한테 건강 전도사 해도 될 수준이다 🔥 계속 가보자!"

        st.markdown("### 🧠 요약 코멘트")
        st.success(msg)
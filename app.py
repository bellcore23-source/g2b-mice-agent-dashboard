import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from openai import OpenAI

# 1. 페이지 기본 설정 및 API 키 로드
st.set_page_config(page_title="MICE 공고 스카우트 v2", page_icon="💼", layout="wide")

G2B_API_KEY = st.secrets["G2B_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# 2. 나라장터 API 호출 함수 (검색일 기준 최근 180일 범위 탐색)
def fetch_g2b_data(keyword):
    url = "http://apis.data.go.kr/1230000/BidPublicInfoService05/getBidPblancListInfoServc05"
    
    # 180일 전 날짜와 오늘 날짜 계산 (YYYYMMDDHHMM 포맷 형태 요구 대응)
    today = datetime.now()
    six_months_ago = today - timedelta(days=180)
    
    str_start_dt = six_months_ago.strftime("%Y%m%d0000")
    str_end_dt = today.strftime("%Y%m%d2359")
    
    params = {
        'serviceKey': G2B_API_KEY,
        'bidNtceNm': keyword,
        'inqryBgnDt': str_start_dt,    # 180일 전부터
        'inqryEndDt': str_end_dt,      # 오늘까지 공고 대상
        'numOfRows': '100',            # 넉넉하게 수집
        'pageNo': '1',
        'inqryDiv': '1',
        'type': 'json'
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        items = res.json().get('response', {}).get('body', {}).get('items', [])
        return items if isinstance(items, list) else []
    except:
        return []

# 3. LLM 에이전트의 맥락 분석 (노이즈 필터링) 함수
def agent_filter_proposals(df_json):
    prompt = f"""
    당신은 MICE 및 학술·비즈니스 행사 대행 공고 선별 에이전트입니다.
    아래 JSON 데이터는 나라장터에서 수집한 '최근 180일간의 용역' 공고 목록입니다.
    이 중 명칭만 관련 키워드가 포함되어 있고 실제로는 '단순 가구/집기 구매', '인쇄물 제작', '장비 대여/리스 및 인프라 구축', '단순 청소/대관' 등 
    행사를 직접 기획하고 총괄 운영하는 대행 용역과 무관한 '물품/인프라성 공고'를 찾아내어 제외(False) 처리하세요.
    실제 행사를 기획하고 총괄 대행하는 용역인 것만 골라내야 합니다.

    [데이터]
    {df_json}

    [출력 포맷]
    반드시 원본 데이터의 '공고번호'를 key로 하고, 유지 여부를 true/false로 판별한 JSON 객체만 반환하세요. 다른 설명은 생략하세요.
    예: {{"20260101-00": true, "20260102-00": false}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ─── UI 레이아웃 및 프론트엔드 시작 ───
st.title("💼 나라장터 MICE 에이전트 검색 시스템 v2")
st.caption("최근 180일 동안 공고된 국제회의, 포럼, 콘퍼런스, 컨퍼런스, 세미나, 회의 용역 공고를 AI 에이전트가 완벽하게 추려냅니다. (금액 제한 없음)")

# 확장된 키워드 리스트
keywords = ["국제회의", "포럼", "콘퍼런스", "컨퍼런스", "세미나", "회의"]

if st.button("🔄 최신 공고 수집 및 에이전트 분석 시작", type="primary"):
    with st.spinner("나라장터 API 및 180일 기간 데이터를 조회하고 에이전트가 맥락을 분석 중입니다..."):
        
        # 1. 원본 데이터 수집
        raw_items = []
        for kw in keywords:
            raw_items.extend(fetch_g2b_data(kw))
            
        if not raw_items:
            st.warning("조건에 맞는 원본 공고 데이터가 나라장터에서 조회되지 않았습니다. API 인증키(Secrets) 설정을 다시 점검해 주세요.")
        else:
            # 2. 1차 정량 필터링 (중복 제거 및 금액 필터 해제)
            refined_list = []
            seen_ids = set()
            
            for item in raw_items:
                bid_id = item.get('bidNtceNo')
                if bid_id in seen_ids: continue
                
                budget_str = item.get('asignBdgtAmt', '0')
                budget = int(float(budget_str)) if budget_str else 0
                
                seen_ids.add(bid_id)
                refined_list.append({
                    "공고번호": bid_id,
                    "공고명": item.get('bidNtceNm'),
                    "수요기관": item.get('dminsttNm'),
                    "배정예산": budget,
                    "마감일시": item.get('bidClseDt'),
                    "URL": item.get('bidNtceDtlUrl')
                })
            
            if not refined_list:
                st.info("조건에 부합하는 공고가 존재하지 않습니다.")
            else:
                # 3. 2차 AI 에이전트 맥락 필터링
                df = pd.DataFrame(refined_list)
                llm_input = df[['공고번호', '공고명']].to_json(orient='records', ensure_ascii=False)
                
                filter_result = agent_filter_proposals(llm_input)
                
                # 에이전트 판별 결과 매핑 및 필터링 적용
                df['에이전트_통과'] = df['공고번호'].map(filter_result)
                final_df = df[df['에이전트_통과'] == True].drop(columns=['에이전트_통과'])
                
                # 배정예산 기준 내림차순 정렬 (높은 순서대로)
                final_df = final_df.sort_values(by="배정예산", ascending=False)
                
                # 4. 결과 테이블 출력
                st.success(f"🤖 분석 완료! 최근 180일 공고 중 노이즈를 제거하고 총 {len(final_df)}건의 핵심 용역 공고를 발굴했습니다.")
                
                # 사용자가 보기 편하도록 금액 포맷팅
                final_df['배정예산'] = final_df['배정예산'].apply(lambda x: f"{x:,}원" if x > 0 else "지급금액미정(확인필요)")
                
                # Streamlit Dataframe 렌더링 (클릭 가능한 하이퍼링크 활성화)
                st.data_editor(
                    final_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("공고 링크", display_text="바로가기")
                    },
                    disabled=True,
                    use_container_width=True
                )

# ✅ 1. 라이브러리 설치
!pip install transformers pandas -q

# ✅ 2. 라이브러리 불러오기
from transformers import pipeline
import pandas as pd
import numpy as np
from datetime import datetime
from google.colab import files

# ✅ 3. 파일 업로드
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# ✅ 4. 날짜 정제
df["댓글 작성일"] = pd.to_datetime(df["댓글 작성일"], errors="coerce")
df = df.dropna(subset=["댓글", "댓글 작성일"])
df["시간대"] = df["댓글 작성일"].dt.floor("H")

# ✅ 5. 감정 분석 모델 로딩
sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ✅ 6. 감정 매핑 함수 (2분류: 긍정/부정)
def map_binary_sentiment(text):
    try:
        label = sentiment_pipe(text[:512])[0]['label']
        if "1" in label or "2" in label:
            return "부정"
        elif "4" in label or "5" in label:
            return "긍정"
        else:
            return "중립 제외"
    except:
        return "오류"

# ✅ 7. 감정 분석 수행
sample = df.head(100).copy()  # 전체 사용 시 .head 제거
sample["댓글 감정"] = sample["댓글"].apply(map_binary_sentiment)

# ✅ 8. 유효한 데이터만 필터링
result = sample[sample["댓글 감정"].isin(["긍정", "부정"])].copy()

# ✅ 9. 시간대별 긍정률 계산
grouped = result.groupby("시간대")["댓글 감정"].value_counts().unstack().fillna(0)
grouped["총 댓글 수"] = grouped.sum(axis=1)
grouped["긍정률"] = (grouped["긍정"] / grouped["총 댓글 수"]) * 100

# ✅ 10. 결과 저장
grouped.reset_index(inplace=True)
grouped.to_csv("시간대별_긍정률_분석결과.csv", index=False, encoding='utf-8-sig')
files.download("시간대별_긍정률_분석결과.csv")

# ✅ 11. 미리보기 출력
print("✅ 시간대별 긍정률 결과")
print(grouped[["시간대", "긍정", "부정", "총 댓글 수", "긍정률"]].round(2))

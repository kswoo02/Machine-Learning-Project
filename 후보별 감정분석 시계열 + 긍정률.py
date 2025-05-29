# ✅ 1. 라이브러리 설치
!pip install transformers pandas tqdm -q

# ✅ 2. 라이브러리 임포트
import pandas as pd
from transformers import pipeline
from datetime import datetime
from google.colab import files
from tqdm.notebook import tqdm

# ✅ 3. 파일 업로드
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# ✅ 4. 날짜 처리 및 시간대 생성
df["댓글 작성일"] = pd.to_datetime(df["댓글 작성일"], errors="coerce")
df = df.dropna(subset=["댓글", "댓글 작성일"])
df["시간대"] = df["댓글 작성일"].dt.floor("H")

# ✅ 5. 감정 분석 모델 로딩
sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ✅ 6. 감정 분석 수행 (진행률 포함)
labels = []
for text in tqdm(df["댓글"], desc="감정 분석 진행 중"):
    try:
        label = sentiment_pipe(text[:512])[0]['label']
        if "1" in label or "2" in label:
            labels.append("부정")
        elif "4" in label or "5" in label:
            labels.append("긍정")
        else:
            labels.append("중립 제외")
    except:
        labels.append("오류")

df["댓글 감정"] = labels

# ✅ 7. 유효한 감정 데이터 필터링
result = df[df["댓글 감정"].isin(["긍정", "부정"])].copy()

# ✅ 8. 시간대별 긍정률 계산
grouped = result.groupby("시간대")["댓글 감정"].value_counts().unstack().fillna(0)
grouped["총 댓글 수"] = grouped.sum(axis=1)
grouped["긍정률"] = (grouped["긍정"] / grouped["총 댓글 수"]) * 100

# ✅ 9. 결과 저장
grouped = grouped.reset_index()[["시간대", "긍정", "부정", "총 댓글 수", "긍정률"]]
grouped.to_csv("시간대별_긍정률_결과.csv", index=False, encoding="utf-8-sig")
files.download("시간대별_긍정률_결과.csv")

# ✅ 10. 출력
print("✅ 시간대별 긍정률 결과:")
print(grouped.round(2))



# 코드 시작
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

# CSV 불러오기
file_path = "2025년05월14일_10시11분 Yna_News_Labeled.csv"
df = pd.read_csv(file_path)

# 모델 및 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
#model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large").to(device)
# 교체: pongjin/roberta_with_kornli
#tokenizer = AutoTokenizer.from_pretrained("pongjin/roberta_with_kornli")
#model = AutoModelForSequenceClassification.from_pretrained("pongjin/roberta_with_kornli")

# 3모델 정치인 관련
tokenizer = AutoTokenizer.from_pretrained("mlburnham/deberta-v3-base-polistance-affect-v1.0")
model = AutoModelForSequenceClassification.from_pretrained("mlburnham/deberta-v3-base-polistance-affect-v1.0")

# 예측 함수
def predict_relation(news_title, comment):
    inputs = tokenizer(news_title, comment, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    labels = ['entailment', 'neutral', 'contradiction']
    prediction = labels[probs.argmax().item()]
    return prediction, {label: round(prob.item(), 4) for label, prob in zip(labels, probs[0])}

# 결과 저장
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    title = row["기사 제목"]
    comment = row["댓글"]
    label = row["라벨"]  # 후보
    pred_label, score_dict = predict_relation(title, comment)
    results.append({
        "기사 제목": title,
        "댓글": comment,
        "후보 라벨": label,
        "NLI 판단": pred_label,
        "entailment": score_dict.get("entailment", 0.0),
        "neutral": score_dict.get("neutral", 0.0),
        "contradiction": score_dict.get("contradiction", 0.0)
    })

# 저장
result_df = pd.DataFrame(results)
result_df.to_csv("news_comment_nli_result.csv", index=False, encoding="utf-8-sig")
print("결과 저장 완료: news_comment_nli_result.csv")

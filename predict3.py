import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
# CSV 불러오기
file_path = "2025년05월25일_16시40분_중앙_이재명_News_Labeled.csv"
df = pd.read_csv(file_path)

# 모델 및 토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("pongjin/roberta_with_kornli")
model = AutoModelForSequenceClassification.from_pretrained("pongjin/roberta_with_kornli").to(device)

# 모델 라벨 확인
print("모델 라벨 정보:", model.config.id2label)

# 예측 함수: 뉴스 내용을 premise, 댓글+후보라벨을 hypothesis 문장으로 활용
def candidate_mentioned(comment, candidate_name):
    # 후보 이름이 단독으로 있거나 조사/특수문자/띄어쓰기와 함께 등장하는 경우 모두 포착
    pattern = rf"\b{re.escape(candidate_name)}([은는이가를의께서]?|\b)"
    return re.search(pattern, comment, flags=re.IGNORECASE) is not None

def predict_relation(premise, comment, candidate_name):
    premise = f"{title} {content}"
    hypothesis = comment
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    labels = ['entailment', 'neutral', 'contradiction']
    prediction = labels[probs.argmax().item()]
    # 🔧 score_dict 먼저 만들고
    score_dict = {label: round(prob.item(), 4) for label, prob in zip(labels, probs[0])}
    print(f"Scores: entailment={score_dict['entailment']}, neutral={score_dict['neutral']}, contradiction={score_dict['contradiction']}")
    print(f"DEBUG: prediction={prediction}, candidate_mentioned={candidate_mentioned(comment, candidate_name)}")
    
    #if prediction == "contradiction" and candidate_mentioned(comment, candidate_name):
    #    if probs[0][2] < 0.7:
    #        prediction = "entailment (후처리)"
    
    return prediction, {label: round(prob.item(), 4) for label, prob in zip(labels, probs[0])}



# 결과 저장
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    title = str(row.get("기사 제목", ""))
    content = str(row.get("기사 본문", ""))
    comment = str(row.get("댓글", ""))
    candidate_label = str(row.get("라벨", ""))
    
     # 날짜 컬럼
    article_date = row.get("기사 작성일", "")
    comment_date = row.get("댓글 작성일", "")

    premise = title + " " + content
    pred_label, score_dict = predict_relation(premise, comment, candidate_label)

    results.append({
        "기사 제목": title,
        "기사 본문": content,
        "기사 작성일": article_date,
        "댓글": comment,
        "댓글 작성일": comment_date,
        "라벨": candidate_label,
        "NLI 판단": pred_label,
        "entailment": score_dict.get("entailment", 0.0),
        "neutral": score_dict.get("neutral", 0.0),
        "contradiction": score_dict.get("contradiction", 0.0)
    })

# 저장
result_df = pd.DataFrame(results)
result_df.to_csv("news_comment_nli_result중앙_이재명샘플.csv", index=False, encoding="utf-8-sig")
print("결과 저장 완료: news_comment_nli_result.csv")

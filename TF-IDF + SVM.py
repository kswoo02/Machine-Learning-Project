import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ✅ 1. CSV 파일 불러오기
df = pd.read_csv("감정분석샘플데이터.csv", encoding="utf-8-sig")

# ✅ 2. 텍스트와 레이블 분리
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

# ✅ 3. 학습용/테스트용 데이터 분리
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ✅ 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# ✅ 5. SVM 모델 학습
model = LinearSVC()
model.fit(X_train, y_train)

# ✅ 6. 예측
y_pred = model.predict(X_test)

# ✅ 7. 결과 출력
print("=== 테스트 문장 예시 ===")
for text in X_test_texts[:5]:
    print(f"- {text}")

print("\n=== 예측 결과 예시 ===")
for text, label in zip(X_test_texts[:5], y_pred[:5]):
    result = "긍정" if label == 1 else "부정"
    print(f"'{text}' → {result}")

# ✅ 8. 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["부정", "긍정"], zero_division=0)

print(f"\n정확도: {accuracy:.2f}")
print(report)

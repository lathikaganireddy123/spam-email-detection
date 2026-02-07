# =========================
# Spam Email Detection
# =========================

import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords (first time only)
nltk.download('stopwords')

# -------------------------
# 1. Load Dataset
# -------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# -------------------------
# 2. Text Preprocessing
# -------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['clean_message'] = data['message'].apply(clean_text)

print("Sample cleaned data:")
print(data[['label', 'message', 'clean_message']].head())

# -------------------------
# 3. Feature Extraction (TF-IDF)
# -------------------------
X = data['clean_message']
y = data['label']

tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(X)

# -------------------------
# 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Model Training (Naive Bayes)
# -------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------
# 6. Model Evaluation
# -------------------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# 7. Save Model
# -------------------------
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("\nModel saved successfully")

# -------------------------
# 8. Prediction Function
# -------------------------
def predict_spam(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    return "Spam" if model.predict(vector)[0] == 1 else "Not Spam"

# -------------------------
# 9. Test with Custom Inputs
# -------------------------
print("\nSingle message test:")
print(predict_spam("Congratulations you won free money"))
print(predict_spam("Hey bro are you coming today"))

print("\nMultiple message tests:")
test_messages = [
    "Congratulations! You have won a free lottery. Click now.",
    "Hey bro, are you coming to college today?",
    "Win a brand new iPhone today! Limited offer.",
    "Please submit the assignment by tonight.",
    "Earn money from home easily. Register now."
]

for msg in test_messages:
    print(msg, "->", predict_spam(msg))

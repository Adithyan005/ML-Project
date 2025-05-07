# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load datasets
true_df = pd.read_csv('datasets\True.csv')
fake_df = pd.read_csv('datasets\Fake.csv')

# Step 3: Add labels
true_df['label'] = 1  # Real news
fake_df['label'] = 0  # Fake news

# Step 4: Combine datasets
data = pd.concat([true_df, fake_df], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Step 5: Preprocessing
# We'll just use the 'text' column for simplicity
X = data['text']
y = data['label']

# Step 6: Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = tfidf.fit_transform(X)

# Step 7: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 9: Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Prediction function
def predict_news(news_text):
    text_vec = tfidf.transform([news_text])
    prediction = model.predict(text_vec)
    return "Real" if prediction[0] == 1 else "Fake"

# Example
sample=input("Enter the news: ")
print("Sample News Prediction:", predict_news(sample))

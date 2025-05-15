import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', names=["text", "emotion"])
    return df

# Train the model
def train_model(data):
    X = data['text']
    y = data['emotion']

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

# Predict emotion from user input
def predict_emotion(model, user_input):
    return model.predict([user_input])[0]

if __name__ == "__main__":
    print("🔄 Training model...")
    df = load_data("train.txt")
    model = train_model(df)
    joblib.dump(model, "emotion_model.pkl")
    print("✅ Model trained and saved!")

    while True:
        user_input = input("\n🗣️ Enter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        prediction = predict_emotion(model, user_input)
        print(f"🔮 Predicted emotion: {prediction}")

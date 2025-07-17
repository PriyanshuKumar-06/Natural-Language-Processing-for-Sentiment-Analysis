import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def create_data(self):
        texts = [
            "I love this movie it is amazing",
            "This product is terrible and awful",
            "The service was okay nothing special",
            "Best purchase ever highly recommend",
            "Waste of money very disappointed",
            "Good quality for the price",
            "Not satisfied with this product",
            "Excellent customer service",
            "Average product nothing exciting",
            "Hate this so much",
            "Really happy with my purchase",
            "Mediocre quality could be better",
            "Outstanding results love it",
            "Poor quality broke quickly",
            "Pretty good overall satisfied",
            "Worst experience ever had",
            "Nice design works well",
            "Decent but not great",
            "Amazing quality exceeded expectations",
            "Terrible service very rude staff"
        ]
        
        labels = [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'positive', 'negative', 'positive', 'neutral', 'negative',
            'positive', 'neutral', 'positive', 'negative', 'positive',
            'negative', 'positive', 'neutral', 'positive', 'negative'
        ]
        
        return pd.DataFrame({'text': texts, 'sentiment': labels})
    
    def train(self, df):
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        X = df['clean_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.model.fit(X_train_vec, y_train)
        
        y_pred = self.model.predict(X_test_vec)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred)
        self.is_trained = True
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def predict(self, text):
        if not self.is_trained:
            print("Please train the model first")
            return None
            
        clean_text = self.clean_text(text)
        text_vec = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        return prediction, probability
    
    def show_results(self, df):
        sentiment_counts = df['sentiment'].value_counts()
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        
        plt.subplot(1, 2, 2)
        plt.bar(sentiment_counts.index, sentiment_counts.values)
        plt.title('Sentiment Counts')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def test_examples(self):
        examples = [
            "I love this product",
            "This is terrible",
            "It's okay",
            "Best thing ever",
            "Not good at all"
        ]
        
        for example in examples:
            sentiment, prob = self.predict(example)
            max_prob = max(prob)
            print(f"'{example}' -> {sentiment} (confidence: {max_prob:.2f})")

def main():
    print("Simple Sentiment Analysis")
    print("=" * 30)
    
    analyzer = SimpleSentimentAnalyzer()
    
    print("Creating training data...")
    df = analyzer.create_data()
    print(f"Data created with {len(df)} examples")
    
    print("\nTraining model...")
    X_train, X_test, y_train, y_test, y_pred = analyzer.train(df)
    print(f"Model trained! Accuracy: {analyzer.accuracy:.2f}")
    
    print("\nModel Performance:")
    print(analyzer.report)
    
    print("\nTesting examples:")
    analyzer.test_examples()
    
    print("\nShowing data visualization...")
    analyzer.show_results(df)
    
    print("\nTry your own text:")
    while True:
        user_text = input("Enter text (or 'quit' to exit): ")
        if user_text.lower() == 'quit':
            break
        
        sentiment, prob = analyzer.predict(user_text)
        confidence = max(prob)
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()

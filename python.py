#Import pandas
import pandas as pd

#Load the IMDb Dataset
data = pd.read_csv('IMDB Dataset.csv')

#Display the first few rows
data.head()

#Check for missing values
data.isnull().sum()

#Drop missing rows(if there are any)
data.dropna(inplace=True)

#Clean the text data
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the reviews
data['cleaned_reviews'] = data['review'].apply(preprocess_text)

#Train test split
from sklearn.model_selection import train_test_split

# Create X (features) and y (labels)
X = data['cleaned_reviews']
y = data['sentiment']  # Assuming sentiment column has positive/negative/neutral labels

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Vectorize the text data
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer on the training data and transform both train and test data
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

#Train the classifier
from sklearn.linear_model import LogisticRegression

# Initialize the classifier
classifier = LogisticRegression()

# Train the classifier
classifier.fit(X_train_vect, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_vect)

#Evaluate model
from sklearn.metrics import accuracy_score, classification_report

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Make predictions
new_review = "Breaking Bad is the greatest series of al time!"
cleaned_new_review = preprocess_text(new_review)
new_review_vect = vectorizer.transform([cleaned_new_review])

# Predict the sentiment
prediction = classifier.predict(new_review_vect)
print("Sentiment:", prediction[0])

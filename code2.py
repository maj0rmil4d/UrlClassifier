import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import joblib
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Load the dataset
df = pd.read_csv('data.csv')

# Extract features and target variable
X = df['cleaned_website_text']
y = df['Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a text classification model using Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
classification_report_str = metrics.classification_report(y_test, y_pred)
print(classification_report_str)

# Save the model for future use
joblib.dump(model, 'website_classifier_model.joblib')

# Function to predict the category of a given URL with a timeout
def predict_category(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout, verify=False, allow_redirects=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        website_text = soup.get_text()
    except requests.exceptions.Timeout:
        print(f"Timeout error for {url}: Request timed out.")
        return None
    except requests.exceptions.RequestException as req_error:
        print(f"Request error for {url}: {req_error}")
        return None
    except Exception as e:
        print(f"Error for {url}: {e}")
        return None

    # Load the pre-trained model
    loaded_model = joblib.load('website_classifier_model.joblib')

    # Make predictions
    category = loaded_model.predict([website_text])[0]
    return category

# Function to read URLs from a file
def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Example usage
file_path = 'urls.txt'
output_file_path = 'urls_classified.txt'
model_path = 'website_classifier_model.joblib'
max_threads = 150
timeout = 10

# Read URLs from file
urls_to_predict = read_urls_from_file(file_path)

# Open file for writing
with open(output_file_path, 'w') as output_file:
    # Use ThreadPoolExecutor with a maximum of 150 threads
    with ThreadPoolExecutor(max_threads) as executor:
        # Iterate over URLs and perform predictions concurrently with a timeout
        futures = [executor.submit(predict_category, url, timeout) for url in urls_to_predict]

        for future, url in zip(concurrent.futures.as_completed(futures), urls_to_predict):
            predicted_category = future.result()
            if predicted_category:
                output_str = f"The predicted category for {url} is: {predicted_category}\n"
                print(output_str)
                output_file.write(output_str)
            else:
                print(f"Failed to predict category for {url}.\n")

# Generate statistics of counts for each category
category_counts = y.value_counts()
print("\nCategory Counts:")
print(category_counts)

# Save statistics to a file
category_counts.to_csv('category_counts.csv')


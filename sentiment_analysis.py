import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the required VADER lexicon dataset from NLTK
# This only runs once to download the rules dictionary
nltk.download('vader_lexicon', quiet=True)

# 1. Create a simulated dataset of product reviews (mimicking Amazon/Social Media data)
data = {
    'Review_ID': [1, 2, 3, 4, 5],
    'Product': ['Wireless Headphones', 'Smart Watch', 'Phone Case', 'Laptop', 'Coffee Maker'],
    'Review_Text': [
        "Absolutely amazing sound quality! I love these headphones so much, highly recommend.",
        "The watch looks nice, but the battery life is incredibly disappointing. It dies in 4 hours.",
        "It is an okay product. Nothing spectacular, just a basic phone case that does the job.",
        "Terrible customer service and the screen arrived cracked. Do not buy this garbage!",
        "Perfect build quality, works like a charm. Very happy with this purchase."
    ]
}

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# 2. Initialize the VADER Sentiment Intensity Analyzer tool
sia = SentimentIntensityAnalyzer()

# 3. Create helper functions to calculate scores and categories
def get_sentiment_scores(text):
    # returns a dictionary with negative, neutral, positive, and compound values
    return sia.polarity_scores(text)['compound']

def categorize_sentiment(compound_score):
    # Standard NLP thresholds for classifying compound metric states
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# 4. Apply the sentiment calculations to our review text columns
df['Compound_Score'] = df['Review_Text'].apply(get_sentiment_scores)
df['Sentiment_Result'] = df['Compound_Score'].apply(categorize_sentiment)

# 5. Display the final categorized dataset output
print("--- TASK 4: SENTIMENT ANALYSIS REPORT ---")
print("\nProcessed Data View:")
for index, row in df.iterrows():
    print(f"\nProduct: {row['Product']}")
    print(f"Review: \"{row['Review_Text']}\"")
    print(f"Analysis Score: {row['Compound_Score']} | Verdict: {row['Sentiment_Result']}")
    print("-" * 50)

# 6. Save the final custom analysis results dataset to a CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
print("\nAnalytics report successfully saved to 'sentiment_analysis_results.csv'!")
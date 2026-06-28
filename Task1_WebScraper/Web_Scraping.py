import requests
from bs4 import BeautifulSoup
import pandas as pd

# 1. Define the target URL
URL = "http://books.toscrape.com/"

# 2. Send an HTTP GET request to the website
response = requests.get(URL)

# Check if the request was successful (Status Code 200)
if response.status_code == 200:
    print("Successfully connected to the website!")
    
    # 3. Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Initialize a list to hold our extracted data
    books_data = []
    
    # 4. Find all book containers on the page
    # Looking through the HTML, each book is stored inside an <article class="product_pod">
    books = soup.find_all('article', class_='product_pod')
    
    for book in books:
        # Extract the title (stored inside the 'title' attribute of the <a> tag inside an <h3>)
        title = book.h3.a['title']
        
        # Extract the price (stored inside a <p class="price_color">)
        price = book.find('p', class_='price_color').text
        
        # Extract availability (stored inside a <p class="instock availability">)
        # .strip() removes unwanted whitespace/newlines
        availability = book.find('p', class_='instock availability').text.strip()
        
        # Append the extracted data as a dictionary to our list
        books_data.append({
            'Title': title,
            'Price': price,
            'Availability': availability
        })
    
    # 5. Convert the list of dictionaries into a structured Pandas DataFrame
    df = pd.DataFrame(books_data)
    
    # 6. Save the custom dataset to a CSV file
    df.to_csv('scraped_books.csv', index=False)
    print("Data successfully scraped and saved to 'scraped_books.csv'!")

else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
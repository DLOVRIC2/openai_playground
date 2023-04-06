import requests
from bs4 import BeautifulSoup
import sqlite3


def get_articles(site):

    if site == "MIT":
        url = "https://news.mit.edu/topic/artificial-intelligence2"

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        articles = soup.find("div", class_="page-term--views--list").contents
        articles = [x for x in articles if type(x).__name__ == "Tag"]

        for article in articles:

            article_url = url.split("/topic")[0] + article.find("a", class_="term-page--news-article--item--title--link").get("href")
            # Send a request to the article URL and parse its content
            article_response = requests.get(article_url)
            article_soup = BeautifulSoup(article_response.content, "html.parser")

            x = 5



    return True


get_articles("MIT")
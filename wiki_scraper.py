import requests
from bs4 import BeautifulSoup
import os

WIKI_BASE_URL = "https://riskofrain2.fandom.com"
START_PAGE = "/wiki/Risk_of_Rain_2_Wiki"  # This is the main wiki page

def scrape_wiki(start_url=START_PAGE, max_pages=5):
    """
    A simple BFS approach starting from the main wiki page, 
    scraping up to `max_pages` pages.
    """
    to_visit = [start_url]
    visited = set()
    pages_content = {}

    while to_visit and len(visited) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)

        full_url = WIKI_BASE_URL + current
        try:
            response = requests.get(full_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text from article body
                article_body = soup.find('div', class_='mw-parser-output')
                text = article_body.get_text(separator='\n') if article_body else ""
                pages_content[current] = text

                # Collect links to other wiki pages
                for link in article_body.find_all('a', href=True):
                    href = link['href']
                    # Only consider /wiki/... links for BFS
                    if href.startswith("/wiki/") and not href.startswith("/wiki/Special"):
                        to_visit.append(href)
        except Exception as e:
            print(f"Error scraping {full_url}: {e}")

    return pages_content

if __name__ == "__main__":
    data = scrape_wiki()
    # Save the scraped data locally
    with open("wiki_data.txt", "w", encoding="utf-8") as f:
        for page_url, page_text in data.items():
            f.write(f"---PAGE: {page_url}---\n{page_text}\n\n")
    print("Wiki data saved to wiki_data.txt")

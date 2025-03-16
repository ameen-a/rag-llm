import logging
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


def clean_html(html_content: str) -> str:
    """Clean HTML content by removing tags and normalizing whitespace"""

    if not html_content:
        return ""

    # parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # extract text content
    text = soup.get_text(separator=" ", strip=True)

    # normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_article_body(html_content: str) -> str:
    """Extract the article-body class content from HTML"""
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # locate article-body div where the article content exists
    article_body = soup.find("div", class_="article-body")

    if article_body:
        return clean_html(str(article_body))
    else:
        logger.warning("Could not find article body in HTML")
        return clean_html(html_content)

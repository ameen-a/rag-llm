import logging
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

def clean_html(html_content: str) -> str:
    """
    Clean HTML content by removing tags and normalizing whitespace.
    
    Args:
        html_content: Raw HTML content to clean
        
    Returns:
        Cleaned text content
    """
    if not html_content:
        return ""
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract text content
    text = soup.get_text(separator=' ', strip=True)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_article_body(html_content: str) -> str:
    """
    Extract specifically the 'article-body' class content from HTML.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Text content of the article body
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the article-body div
    article_body = soup.find('div', class_='article-body')
    
    if article_body:
        return clean_html(str(article_body))
    else:
        logger.warning("Could not find article-body class in HTML")
        return clean_html(html_content)
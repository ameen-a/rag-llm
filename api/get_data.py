import logging
import time
import requests
import json
import os
from typing import Dict, List, Any
from .utils import clean_html

logger = logging.getLogger(__name__)


class VoyZendeskAPI:
    """Client for extracting API data from Zendesk Help Center"""

    BASE_URL = "https://joinvoy.zendesk.com/api/v2/help_center/en-gb"

    def __init__(self, rate_limit_delay: float = 0.5):
        self.rate_limit_delay = rate_limit_delay
        logger.info("Initialised Zendesk API class")

    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request to API endpoint, with rate limiting"""
        url = f"{self.BASE_URL}/{endpoint}"
        logger.debug(f"Making request to {url}")

        # pro-active rate limiting
        time.sleep(self.rate_limit_delay)

        max_retries = 3
        retry_count = 0
        backoff_time = 1.0

        while retry_count <= max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()

                return response.json()

            except requests.exceptions.RequestException as e:
                retry_count += 1

                if retry_count > max_retries:
                    logger.error(
                        f"Failed to fetch {url} after {max_retries} attempts: {str(e)}"
                    )
                    raise

                logger.warning(
                    f"Request failed, retrying in {backoff_time:.1f}s (Attempt {retry_count}/{max_retries})"
                )
                time.sleep(backoff_time)
                backoff_time *= 2  # exponential backoff for retrying failed requests

        raise RuntimeError("Request failed with unknown error")

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Fetch FAQ categories"""
        logger.info("Fetching all categories")
        response = self._make_request("categories.json")
        return response.get("categories", [])

    def get_sections_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        """Fetch sections for a given category"""
        logger.info(f"Fetching sections for category {category_id}")
        response = self._make_request(f"categories/{category_id}/sections.json")
        return response.get("sections", [])

    def get_article(self, article_id: int) -> Dict[str, Any]:
        """Fetch a specific article by ID"""
        logger.info(f"Fetching article {article_id}")
        response = self._make_request(f"articles/{article_id}.json")
        return response.get("article", {})

    def get_articles_by_section(self, section_id: int) -> List[Dict[str, Any]]:
        """Fetch all articles for a given section"""
        logger.info(f"Fetching articles for section {section_id}")
        response = self._make_request(f"sections/{section_id}/articles.json")
        return response.get("articles", [])

    def extract_all_articles(
        self, save_raw: bool = True, raw_dir: str = "../data/raw"
    ) -> List[Dict[str, Any]]:
        """Extract all articles from the Zendesk API"""

        logger.info("Beginning article extraction")

        if save_raw:
            # store raw data here
            os.makedirs(raw_dir, exist_ok=True)

        all_articles = []

        # step 1: collect all article IDs by traversing the categories and sections
        article_ids = []
        categories = self.get_all_categories()

        for category in categories:
            category_id = category["id"]
            category_name = category["name"]

            sections = self.get_sections_by_category(category_id)

            for section in sections:
                section_id = section["id"]
                section_name = section["name"]

                article_listings = self.get_articles_by_section(section_id)

                for article_listing in article_listings:
                    article_ids.append(
                        {
                            "article_id": article_listing["id"],
                            "category_id": category_id,
                            "category_name": category_name,
                            "section_id": section_id,
                            "section_name": section_name,
                            "title": article_listing.get("title", ""),
                        }
                    )

        # step 2: fetch each article's data
        logger.info(f"Found {len(article_ids)} articles to fetch")
        for idx, article_meta in enumerate(article_ids):
            article_id = article_meta["article_id"]
            logger.info(f"Fetching article {idx+1}/{len(article_ids)}: {article_id}")

            article = self.get_article(article_id)

            # step 3: save and process article data
            if save_raw:
                with open(f"{raw_dir}/article_{article_id}.json", "w") as f:
                    json.dump(article, f, indent=2)

            article_body = article.get("body", "")
            cleaned_body = clean_html(article_body)

            processed_article = {  # create structured article data object
                "id": article_id,
                "title": article.get("title", ""),
                "body": cleaned_body,
                "html_body": article_body,
                "url": article.get("html_url", ""),
                "category": {
                    "id": article_meta["category_id"],
                    "name": article_meta["category_name"],
                },
                "section": {
                    "id": article_meta["section_id"],
                    "name": article_meta["section_name"],
                },
                "tags": article.get("label_names", []),
                "created_at": article.get("created_at", ""),
                "updated_at": article.get("updated_at", ""),
            }

            all_articles.append(processed_article)

        return all_articles

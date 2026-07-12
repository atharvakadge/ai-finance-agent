"""
News Service - Fetches latest financial news via Google News RSS.
No API key needed. Free. Reliable.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


class NewsService:

    def search_news(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"https://news.google.com/rss/search?"
                f"q={encoded_query}+financial&hl=en-IN&gl=IN&ceid=IN:en"
            )

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FinLensAI/1.0"},
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read().decode()

            root = ET.fromstring(xml_data)
            articles = []

            for item in root.findall(".//item")[:max_results]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                source = item.findtext("source", "Unknown")

                articles.append({
                    "title": title,
                    "snippet": f"Published: {pub_date}",
                    "url": link,
                    "source": source,
                })

            if not articles:
                articles.append({
                    "title": f"No recent news found for '{query}'",
                    "snippet": "Try a different search term.",
                    "url": "",
                    "source": "none",
                })

            return articles

        except Exception as e:
            return [{
                "title": "News fetch failed",
                "snippet": f"Could not retrieve news: {str(e)}",
                "url": "",
                "source": "error",
            }]

    def get_company_news(self, company_name: str) -> dict:
        articles = self.search_news(company_name)

        return {
            "company": company_name,
            "articles": articles,
            "count": len(articles),
        }
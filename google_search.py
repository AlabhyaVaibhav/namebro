"""
Google Search Integration for Startup Name Research

Searches Google to find existing applications, companies, and domains
with similar names to provide competitive insights.
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
from urllib.parse import quote_plus

load_dotenv()


@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    snippet: str
    domain: Optional[str] = None


@dataclass
class SearchInsights:
    """Insights from Google search results"""
    query: str
    results: List[SearchResult]
    total_results: int
    domains_found: List[str]
    categories: List[str]
    summary: str
    error: Optional[str] = None


class GoogleSearcher:
    """
    Google search integration using Google Custom Search API.
    Falls back to alternative methods if API is not configured.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 search_engine_id: Optional[str] = None):
        """
        Initialize the Google searcher.
        
        Args:
            api_key: Google Custom Search API key (or from GOOGLE_API_KEY env var)
            search_engine_id: Custom Search Engine ID (or from GOOGLE_SEARCH_ENGINE_ID env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.use_api = bool(self.api_key and self.search_engine_id)
        
        # Base URL for Google Custom Search API
        self.api_url = "https://www.googleapis.com/customsearch/v1"
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return url
    
    def _categorize_result(self, title: str, snippet: str) -> List[str]:
        """
        Categorize a search result based on title and snippet.
        Returns list of categories/domains the result belongs to.
        """
        categories = []
        text = (title + " " + snippet).lower()
        
        # Technology/Software categories
        if any(word in text for word in ["app", "application", "software", "platform", "saas"]):
            categories.append("Software/App")
        if any(word in text for word in ["api", "sdk", "developer", "integration"]):
            categories.append("Developer Tools")
        if any(word in text for word in ["ai", "artificial intelligence", "machine learning", "ml"]):
            categories.append("AI/ML")
        if any(word in text for word in ["cloud", "hosting", "infrastructure", "server"]):
            categories.append("Cloud/Infrastructure")
        
        # Business categories
        if any(word in text for word in ["startup", "company", "business", "enterprise"]):
            categories.append("Business/Company")
        if any(word in text for word in ["service", "solution", "product"]):
            categories.append("Service/Product")
        
        # Domain-specific
        if any(word in text for word in ["domain", "website", "web", "online"]):
            categories.append("Web/Domain")
        if any(word in text for word in ["marketplace", "ecommerce", "shop", "store"]):
            categories.append("E-commerce")
        if any(word in text for word in ["social", "community", "network"]):
            categories.append("Social/Community")
        
        return categories if categories else ["General"]
    
    def search_with_api(self, query: str, num_results: int = 20) -> SearchInsights:
        """
        Search using Google Custom Search API.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve (max 20 per request, can paginate)
            
        Returns:
            SearchInsights object
        """
        if not self.use_api:
            return SearchInsights(
                query=query,
                results=[],
                total_results=0,
                domains_found=[],
                categories=[],
                summary="Google Custom Search API not configured",
                error="API credentials missing"
            )
        
        try:
            # Google Custom Search API allows max 10 results per request
            # For 20 results, we need 2 requests
            all_results = []
            start_index = 1
            
            # Get up to 20 results (2 pages of 10)
            for page in range(2):  # 2 pages = 20 results
                params = {
                    "key": self.api_key,
                    "cx": self.search_engine_id,
                    "q": query,
                    "num": 10,  # Max per request
                    "start": start_index
                }
                
                response = requests.get(self.api_url, params=params, timeout=10)
                
                if response.status_code != 200:
                    error_msg = f"API error: {response.status_code}"
                    if response.status_code == 403:
                        error_msg += " - Check API key and search engine ID"
                    return SearchInsights(
                        query=query,
                        results=[],
                        total_results=0,
                        domains_found=[],
                        categories=[],
                        summary="Search failed",
                        error=error_msg
                    )
                
                data = response.json()
                
                # Parse results
                items = data.get("items", [])
                for item in items:
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        domain=self._extract_domain(item.get("link", ""))
                    )
                    all_results.append(result)
                
                # Check if there are more results
                if "queries" in data and "nextPage" in data["queries"]:
                    start_index += 10
                else:
                    break
                
                # Rate limiting - wait between requests
                if page < 1:  # Don't wait after last request
                    time.sleep(0.5)
            
            # Extract insights
            domains_found = list(set([r.domain for r in all_results if r.domain]))
            categories = []
            for result in all_results:
                cats = self._categorize_result(result.title, result.snippet)
                categories.extend(cats)
            unique_categories = list(set(categories))
            
            # Generate summary
            total_found = data.get("searchInformation", {}).get("totalResults", "0")
            summary = f"Found {len(all_results)} results. "
            if domains_found:
                summary += f"Top domains: {', '.join(domains_found[:5])}. "
            if unique_categories:
                summary += f"Categories: {', '.join(unique_categories[:5])}."
            
            return SearchInsights(
                query=query,
                results=all_results[:num_results],
                total_results=int(total_found.replace(",", "")) if isinstance(total_found, str) else total_found,
                domains_found=domains_found[:10],
                categories=unique_categories[:10],
                summary=summary
            )
            
        except requests.exceptions.RequestException as e:
            return SearchInsights(
                query=query,
                results=[],
                total_results=0,
                domains_found=[],
                categories=[],
                summary="Search request failed",
                error=f"Request error: {str(e)}"
            )
        except Exception as e:
            return SearchInsights(
                query=query,
                results=[],
                total_results=0,
                domains_found=[],
                categories=[],
                summary="Search failed",
                error=f"Unexpected error: {str(e)}"
            )
    
    def search_fallback(self, query: str, num_results: int = 20) -> SearchInsights:
        """
        Fallback search method using DuckDuckGo or similar.
        This is a simple fallback if Google API is not available.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            
        Returns:
            SearchInsights object
        """
        try:
            # Use DuckDuckGo HTML search as fallback
            ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(ddg_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return SearchInsights(
                    query=query,
                    results=[],
                    total_results=0,
                    domains_found=[],
                    categories=[],
                    summary="Fallback search unavailable",
                    error="HTTP error"
                )
            
            # Simple parsing (this is a basic fallback)
            # For production, consider using a proper library
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            result_divs = soup.find_all("div", class_="result")[:num_results]
            
            for div in result_divs:
                title_elem = div.find("a", class_="result__a")
                snippet_elem = div.find("a", class_="result__snippet")
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get("href", "")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        domain=self._extract_domain(url)
                    ))
            
            domains_found = list(set([r.domain for r in results if r.domain]))
            categories = []
            for result in results:
                cats = self._categorize_result(result.title, result.snippet)
                categories.extend(cats)
            unique_categories = list(set(categories))
            
            summary = f"Found {len(results)} results via fallback search. "
            if domains_found:
                summary += f"Domains: {', '.join(domains_found[:5])}."
            
            return SearchInsights(
                query=query,
                results=results,
                total_results=len(results),
                domains_found=domains_found[:10],
                categories=unique_categories[:10],
                summary=summary
            )
            
        except ImportError:
            # BeautifulSoup not available
            return SearchInsights(
                query=query,
                results=[],
                total_results=0,
                domains_found=[],
                categories=[],
                summary="Fallback search requires beautifulsoup4",
                error="beautifulsoup4 not installed"
            )
        except Exception as e:
            return SearchInsights(
                query=query,
                results=[],
                total_results=0,
                domains_found=[],
                categories=[],
                summary="Fallback search failed",
                error=str(e)
            )
    
    def search(self, query: str, num_results: int = 20, use_fallback: bool = True) -> SearchInsights:
        """
        Search Google for a query.
        
        Args:
            query: Search query
            num_results: Number of results to retrieve (default 20, max 20 with API)
            use_fallback: Use fallback method if API is not available
            
        Returns:
            SearchInsights object
        """
        if self.use_api:
            return self.search_with_api(query, num_results)
        elif use_fallback:
            return self.search_fallback(query, num_results)
        else:
            return SearchInsights(
                query=query,
                results=[],
                total_results=0,
                domains_found=[],
                categories=[],
                summary="Google search not configured",
                error="API credentials missing and fallback disabled"
            )
    
    def search_startup_name(self, name: str) -> SearchInsights:
        """
        Search for a startup name to find existing applications/companies.
        
        Args:
            name: Startup name to search for
            
        Returns:
            SearchInsights object with competitive analysis
        """
        # Create search queries
        queries = [
            f'"{name}"',  # Exact match
            f"{name} app",  # App search
            f"{name} startup",  # Startup search
            f"{name} company"  # Company search
        ]
        
        all_results = []
        all_domains = set()
        all_categories = set()
        
        for query in queries[:2]:  # Use top 2 queries to get top 2 pages worth of results
            insights = self.search(query, num_results=10)
            if insights.results:
                all_results.extend(insights.results)
                all_domains.update(insights.domains_found)
                all_categories.update(insights.categories)
            
            # Rate limiting between queries
            time.sleep(0.3)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Limit to top results
        unique_results = unique_results[:20]
        
        # Generate comprehensive summary
        summary_parts = []
        summary_parts.append(f"Found {len(unique_results)} unique results for '{name}'.")
        
        if all_domains:
            summary_parts.append(f"Domains found: {', '.join(list(all_domains)[:5])}.")
        
        if all_categories:
            summary_parts.append(f"Categories: {', '.join(list(all_categories)[:5])}.")
        
        if not unique_results:
            summary_parts.append("No existing applications found with this name.")
        
        return SearchInsights(
            query=f'"{name}" (startup/app/company)',
            results=unique_results,
            total_results=len(unique_results),
            domains_found=list(all_domains)[:10],
            categories=list(all_categories)[:10],
            summary=" ".join(summary_parts)
        )


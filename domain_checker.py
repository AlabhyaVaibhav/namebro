"""
Domain Availability Checker using GoDaddy Domains API

Integrates with GoDaddy API to check domain availability and generates
creative domain variations for startup names.
"""

import os
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import requests

load_dotenv()


@dataclass
class DomainResult:
    """Result of a domain availability check"""
    domain: str
    available: bool
    price: Optional[float] = None
    currency: Optional[str] = None
    period: Optional[int] = None
    error: Optional[str] = None


class DomainChecker:
    """
    Domain availability checker using GoDaddy Domains API.
    Supports both OTE (test) and production environments.
    """
    
    # Popular TLDs for startups
    POPULAR_TLDS = [
        'com', 'io', 'ai', 'app', 'dev', 'tech', 'co', 'net', 'org',
        'xyz', 'online', 'cloud', 'digital', 'studio', 'labs', 'space'
    ]
    
    # Creative prefixes and suffixes
    PREFIXES = ['get', 'try', 'use', 'join', 'go', 'my', 'the', 'we']
    SUFFIXES = ['app', 'ai', 'io', 'hub', 'lab', 'ly', 'fy', 'ly', 'co']
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 use_ote: bool = True):
        """
        Initialize the domain checker.
        
        Args:
            api_key: GoDaddy API key (or from GODADDY_API_KEY env var)
            api_secret: GoDaddy API secret (or from GODADDY_API_SECRET env var)
            use_ote: Use OTE (test) environment if True, production if False
        """
        self.api_key = api_key or os.getenv("GODADDY_API_KEY")
        self.api_secret = api_secret or os.getenv("GODADDY_API_SECRET")
        self.use_ote = use_ote
        
        if use_ote:
            self.base_url = "https://api.ote-godaddy.com"
        else:
            self.base_url = "https://api.godaddy.com"
        
        self.session = requests.Session()
        if self.api_key and self.api_secret:
            self.session.headers.update({
                "Authorization": f"sso-key {self.api_key}:{self.api_secret}",
                "Accept": "application/json"
            })
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a name for use in domain names.
        Removes spaces, special characters, and converts to lowercase.
        """
        # Remove common words that might be in names
        name = name.lower().strip()
        # Remove special characters, keep only alphanumeric
        name = re.sub(r'[^a-z0-9]', '', name)
        return name
    
    def generate_domain_variations(self, name: str) -> List[str]:
        """
        Generate creative domain variations for a given name.
        
        Examples:
        - name.app, name.ai, name.io
        - getname.com, tryname.com
        - nameapp.com, nameai.com
        - namehub.com, namelab.com
        
        Args:
            name: The base name to generate variations from
            
        Returns:
            List of domain name variations
        """
        variations = []
        sanitized = self._sanitize_name(name)
        
        if not sanitized:
            return variations
        
        # Basic TLD variations
        for tld in self.POPULAR_TLDS[:10]:  # Limit to top 10 for performance
            variations.append(f"{sanitized}.{tld}")
        
        # Prefix variations (getname.com, tryname.com)
        for prefix in self.PREFIXES[:5]:  # Top 5 prefixes
            variations.append(f"{prefix}{sanitized}.com")
            variations.append(f"{prefix}{sanitized}.app")
            variations.append(f"{prefix}{sanitized}.io")
        
        # Suffix variations (nameapp.com, nameai.com)
        for suffix in self.SUFFIXES[:5]:  # Top 5 suffixes
            variations.append(f"{sanitized}{suffix}.com")
            variations.append(f"{sanitized}{suffix}.app")
            variations.append(f"{sanitized}{suffix}.io")
        
        # Combined variations (getnameapp.com)
        for prefix in self.PREFIXES[:3]:
            for suffix in self.SUFFIXES[:3]:
                variations.append(f"{prefix}{sanitized}{suffix}.com")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations[:30]  # Limit to 30 variations
    
    def check_domain_availability(self, domain: str) -> DomainResult:
        """
        Check if a single domain is available using GoDaddy API.
        
        Args:
            domain: Domain name to check (e.g., "example.com")
            
        Returns:
            DomainResult object with availability status
        """
        if not self.api_key or not self.api_secret:
            return DomainResult(
                domain=domain,
                available=False,
                error="GoDaddy API credentials not configured. Set GODADDY_API_KEY and GODADDY_API_SECRET in .env file"
            )
        
        try:
            url = f"{self.base_url}/v1/domains/available"
            params = {"domain": domain}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                available = data.get("available", False)
                
                # Extract pricing information if available
                price = None
                currency = None
                period = None
                
                if "price" in data:
                    price_info = data["price"]
                    price = price_info.get("salePrice") or price_info.get("listPrice")
                    currency = price_info.get("currency", "USD")
                    period = price_info.get("period", 1)
                
                return DomainResult(
                    domain=domain,
                    available=available,
                    price=price,
                    currency=currency,
                    period=period
                )
            elif response.status_code == 422:
                # Domain format invalid
                return DomainResult(
                    domain=domain,
                    available=False,
                    error="Invalid domain format"
                )
            else:
                return DomainResult(
                    domain=domain,
                    available=False,
                    error=f"API error: {response.status_code} - {response.text[:100]}"
                )
                
        except requests.exceptions.RequestException as e:
            return DomainResult(
                domain=domain,
                available=False,
                error=f"Request failed: {str(e)}"
            )
        except Exception as e:
            return DomainResult(
                domain=domain,
                available=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def check_multiple_domains(self, domains: List[str], delay: float = 0.1) -> List[DomainResult]:
        """
        Check availability for multiple domains with rate limiting.
        
        Args:
            domains: List of domain names to check
            delay: Delay between requests in seconds (to respect rate limits)
            
        Returns:
            List of DomainResult objects
        """
        results = []
        
        for i, domain in enumerate(domains):
            result = self.check_domain_availability(domain)
            results.append(result)
            
            # Rate limiting - wait between requests
            if i < len(domains) - 1:
                time.sleep(delay)
        
        return results
    
    def check_name_variations(self, name: str, max_checks: int = 20) -> Dict[str, List[DomainResult]]:
        """
        Generate domain variations for a name and check their availability.
        
        Args:
            name: Base name to generate variations from
            max_checks: Maximum number of domains to check
            
        Returns:
            Dictionary with 'available' and 'unavailable' lists of DomainResult objects
        """
        variations = self.generate_domain_variations(name)
        
        # Limit the number of checks
        variations_to_check = variations[:max_checks]
        
        results = self.check_multiple_domains(variations_to_check)
        
        available = [r for r in results if r.available and not r.error]
        unavailable = [r for r in results if not r.available and not r.error]
        errors = [r for r in results if r.error]
        
        return {
            "available": available,
            "unavailable": unavailable,
            "errors": errors,
            "total_checked": len(results)
        }
    
    def get_best_available_domains(self, name: str, max_results: int = 5) -> List[DomainResult]:
        """
        Get the best available domain options for a name.
        Prioritizes .com, .io, .app, .ai TLDs.
        
        Args:
            name: Base name
            max_results: Maximum number of results to return
            
        Returns:
            List of available DomainResult objects, sorted by preference
        """
        results = self.check_name_variations(name, max_checks=30)
        available = results["available"]
        
        if not available:
            return []
        
        # Priority order: .com > .io > .app > .ai > others
        priority_tlds = ['com', 'io', 'app', 'ai', 'dev', 'tech', 'co']
        
        def get_priority(domain_result: DomainResult) -> int:
            tld = domain_result.domain.split('.')[-1]
            try:
                return priority_tlds.index(tld)
            except ValueError:
                return 999  # Low priority for non-priority TLDs
        
        # Sort by priority, then by domain length (shorter is better)
        sorted_domains = sorted(
            available,
            key=lambda x: (get_priority(x), len(x.domain))
        )
        
        return sorted_domains[:max_results]
    
    def calculate_domain_availability_score(self, name: str) -> Tuple[int, List[str]]:
        """
        Calculate a domain availability score (1-10) for a name.
        Higher score means better domain availability.
        
        Args:
            name: Name to evaluate
            
        Returns:
            Tuple of (score 1-10, list of available domains)
        """
        results = self.check_name_variations(name, max_checks=15)
        available = results["available"]
        
        if not available:
            return 1, []
        
        # Score based on:
        # - Number of available domains (more = better)
        # - Quality of TLDs (com, io, app, ai are premium)
        # - Domain length (shorter = better)
        
        premium_tlds = ['com', 'io', 'app', 'ai']
        premium_count = sum(1 for r in available if r.domain.split('.')[-1] in premium_tlds)
        
        # Base score from number of available domains
        base_score = min(5, len(available) // 2)
        
        # Bonus for premium TLDs
        premium_bonus = min(3, premium_count)
        
        # Bonus for short domains
        avg_length = sum(len(r.domain) for r in available) / len(available) if available else 20
        length_bonus = max(0, 2 - (avg_length - 10) / 5) if avg_length > 10 else 2
        
        total_score = min(10, int(base_score + premium_bonus + length_bonus))
        
        # Get top available domains
        top_domains = [r.domain for r in available[:5]]
        
        return total_score, top_domains


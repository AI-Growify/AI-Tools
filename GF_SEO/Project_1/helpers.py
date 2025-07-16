import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import requests
from textwrap import wrap
import json
import streamlit as st
import os
from dotenv import load_dotenv
import re
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

_http_head_cache: dict[str, int] = {}   # outbound URL ➜ status code
_session = requests.Session()           # connection reuse

def head_cached(url, timeout=5):
    code = _http_head_cache.get(url)
    if code is None:
        try:
            resp = _session.head(url, timeout=timeout, allow_redirects=True)
            code = resp.status_code
            if code == 405:  # Method Not Allowed
                resp = _session.get(url, timeout=timeout, allow_redirects=True)
                code = resp.status_code
        except Exception:
            code = 599
        _http_head_cache[url] = code
    return code


def head_cached_parallel(urls: list[str], max_workers: int = 12) -> dict[str, int]:
    """
    Returns {url: status_code} for every URL in <urls>.
    Uses cache, runs HEAD calls in parallel where needed.
    """
    results: dict[str, int] = {}

    # ---- split URLs: cached vs not‑cached ------------------------
    to_fetch = []
    for u in urls:
        code = _http_head_cache.get(u)
        if code is not None:
            results[u] = code
        else:
            to_fetch.append(u)

    # ---- fetch the uncached ones in parallel --------------------
    if to_fetch:
        def _hc(u):  # small wrapper so map returns (url, code)
            return u, head_cached(u)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for u, code in pool.map(_hc, to_fetch):
                results[u] = code  # head_cached has already stored it too
    return results


def display_wrapped_json(data: list[dict], preview: int = 10) -> None:
    """
    Show just the first <preview> items of a huge JSON list,
    then offer a download link for the full file.
    """
    if not data:
        st.write("⛔️ No data to display.")
        return

    # ---- preview block ----
    preview_block = data[:preview]
    st.json(preview_block, expanded=False)
    st.caption(f"Showing the first {len(preview_block)} of {len(data)} pages")

    # ---- full‑download button ----
    st.download_button(
        "📥 Download full SEO report (JSON)",
        json.dumps(data, indent=2),
        file_name="seo_report.json",
        mime="application/json",
    )

    
def should_skip_url(url):
    skip_keywords = ["/cart", "/checkout", "/login", "/account", "/my-account"]
    if any(kw in url.lower() for kw in skip_keywords):
        return True
    if "?" in url:  # Skip paginated, sorted, filtered variants
        return True
    return False

def get_rendered_html(url, driver=None, wait=0.5):
    try:
        if driver is None:
            raise ValueError("Pass an existing driver for best speed")

        driver.get(url)
        time.sleep(wait)            # 0.3–0.5 s is usually enough
        return driver.page_source
    except Exception as e:
        print(f"❌ Failed to render {url}: {e}")
        return None


def extract_internal_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    internal_links = set()
    domain = urlparse(base_url).netloc
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href.startswith("/") or domain in href:
            full_url = urljoin(base_url, href)
            internal_links.add(full_url.split("#")[0])
    return list(internal_links)

def get_urls_from_sitemap(sitemap_url):
    """
    Handles both sitemap.xml and sitemap indexes.
    Returns a deduplicated list of on‑site HTML URLs
    (skips images, PDFs, etc.).
    """
    EXT_SKIP = {".jpg", ".jpeg", ".png", ".gif", ".webp",
                ".svg", ".pdf", ".zip", ".mp4", ".mov"}

    def fetch_soup(url):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return BeautifulSoup(r.content, "xml")
        except Exception as e:
            print(f"❌ Failed to fetch sitemap {url}: {e}")
            return None

    def is_html_link(link, base_netloc):
        parsed = urlparse(link)
        if parsed.netloc and parsed.netloc != base_netloc:
            return False
        if any(parsed.path.lower().endswith(ext) for ext in EXT_SKIP):
            return False
        return True

    root_soup = fetch_soup(sitemap_url)
    if not root_soup:
        return []

    base_netloc = urlparse(sitemap_url).netloc
    urls = set()

    if root_soup.find("sitemapindex"):
        for sm in root_soup.find_all("sitemap"):
            loc = sm.find("loc")
            if not loc:
                continue
            child_soup = fetch_soup(loc.text.strip())
            if not child_soup:
                continue
            for url_tag in child_soup.find_all("url"):
                loc_tag = url_tag.find("loc")
                if loc_tag and is_html_link(loc_tag.text.strip(), base_netloc):
                    urls.add(loc_tag.text.strip())
    else:
        for url_tag in root_soup.find_all("url"):
            loc_tag = url_tag.find("loc")
            if loc_tag and is_html_link(loc_tag.text.strip(), base_netloc):
                urls.add(loc_tag.text.strip())

    return list(urls)



def full_seo_audit(url, titles_seen, descs_seen, content_hashes_seen, html):
    """Run the full on‑page technical SEO audit for a single URL.

    *Uses cached + parallel HEAD requests for outbound links, images, and
    internal link‑error checks to maximise speed.*
    """
    result = {}
    visited_urls: set[str] = set()
    internal_errors: list[dict] = []

    try:
        if not html:
            result["error"] = f"Could not render page: {url}"
            return result

        soup = BeautifulSoup(html, "html.parser")
        anchor_tags = soup.find_all("a", href=True)
        parsed_url = urlparse(url)

        # ─── Meta data ────────────────────────────────────────────────
        title_tag = soup.find("title")
        desc_tag = soup.find("meta", {"name": "description"})
        title_text = title_tag.text.strip() if title_tag else ""
        desc_text  = desc_tag.get("content", "").strip() if desc_tag else ""

        result["title"] = {
            "text": title_text or "Missing",
            "length": len(title_text),
            "word_count": len(title_text.split()),
        }
        result["description"] = {
            "text": desc_text or "Missing",
            "length": len(desc_text),
            "word_count": len(desc_text.split()),
        }

        # Duplicate checks
        if title_text in titles_seen:
            result["duplicate_title"] = True
        titles_seen.add(title_text)

        if desc_text in descs_seen:
            result["duplicate_meta_description"] = True
        descs_seen.add(desc_text)

        page_text    = " ".join(soup.stripped_strings)
        text_hash    = hash(page_text)
        if text_hash in content_hashes_seen:
            result["duplicate_content"] = True
        content_hashes_seen.add(text_hash)

        # Headings counts + H1/H1‑title dupes
        result["headings"] = {f"H{i}": len(soup.find_all(f"h{i}")) for i in range(1, 7)}
        h1_tag = soup.find("h1")
        h1_text = h1_tag.text.strip() if h1_tag else ""
        result["H1_content"] = h1_text
        if h1_text and title_text and h1_text.lower() == title_text.lower():
            result["h1_title_duplicate"] = True

        # ─── External (outbound) link checks – Parallel + Cached ─────
        SOCIAL_DOMAINS = [
            "facebook.com", "instagram.com", "twitter.com", "linkedin.com", "x.com",
            "youtube.com", "pinterest.com", "tiktok.com", "wa.me", "web.whatsapp.com"
        ]

        urls_to_test = []
        for a in anchor_tags:
            href = a.get("href")
            if not href:
                continue
            full_url    = urljoin(url, href)
            href_domain = urlparse(full_url).netloc
            if any(s in href_domain for s in SOCIAL_DOMAINS):
                continue  # skip social
            if href_domain and href_domain != parsed_url.netloc:
                urls_to_test.append(full_url)

        external_broken_links = [
            {"url": u, "status": code}
            for u, code in head_cached_parallel(urls_to_test).items()
            if code >= 400
        ]
        result["external_broken_links"] = external_broken_links

        # ─── Text / anchor stats ─────────────────────────────────────
        total_words  = len(re.findall(r"\b\w+\b", page_text))
        anchor_texts = [a.get_text(strip=True) for a in anchor_tags if a.get_text(strip=True)]
        anchor_words = sum(len(t.split()) for t in anchor_texts)

        result["word_stats"] = {
            "total_words": total_words,
            "anchor_words": anchor_words,
            "anchor_ratio_percent": round((anchor_words / total_words) * 100, 2) if total_words else 0,
            "sample_anchors": anchor_texts[:10],
        }
        result["empty_anchor_text_links"] = sum(1 for a in anchor_tags if not a.get_text(strip=True))
        non_desc = {"click here", "read more", "learn more", "more", "here", "view"}
        result["non_descriptive_anchors"] = sum(1 for t in anchor_texts if t.lower() in non_desc)

        # ─── HTTPS & misc checks ─────────────────────────────────────
        result["https_info"] = {"using_https": url.startswith("https://"), "was_redirected": False}
        if len(anchor_tags) <= 1:
            result["single_internal_link"] = True
        http_links = [urljoin(url, a["href"]) for a in anchor_tags if url.startswith("https://") and urljoin(url, a["href"]).startswith("http://")]
        if http_links:
            result["http_links_on_https"] = http_links
        if parsed_url.query:
            result["url_has_parameters"] = True
        html_size = len(html)
        result["text_to_html_ratio_percent"] = round((len(page_text) / html_size) * 100, 2) if html_size else 0
        result["schema"] = {
            "json_ld_found": bool(soup.find_all("script", {"type": "application/ld+json"})),
            "microdata_found": bool(soup.find_all(attrs={"itemscope": True})),
        }

        # ─── Image checks – Parallel + Cached (first 10 images) ─────
        images      = soup.find_all("img")
        image_urls  = [urljoin(url, img.get("src")) for img in images[:10] if img.get("src")]
        broken_images = [
            {"src": u, "status": code}
            for u, code in head_cached_parallel(image_urls).items()
            if code >= 400
        ]
        result["images"] = {
            "total_images": len(images),
            "images_without_alt": sum(1 for img in images if not img.get("alt")),
            "sample_images": [{"src": img.get("src"), "alt": img.get("alt")} for img in images[:5]],
            "broken_images": broken_images,
        }

        # ─── robots.txt quick check ─────────────────────────────────
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        try:
            robots_response = requests.get(robots_url, timeout=5)
            disallows = [ln.strip() for ln in robots_response.text.splitlines() if ln.lower().startswith("disallow")]
            result["robots_txt"] = {"found": True, "disallows": disallows}
        except Exception:
            result["robots_txt"] = {"found": False, "disallows": []}

        meta_robots = soup.find("meta", {"name": "robots"})
        result["meta_robots"] = meta_robots.get("content", "") if meta_robots else ""

        # ─── Internal link error probe – Parallel + Cached ──────────
        base_domain   = parsed_url.netloc
        internal_urls = []
        for a in anchor_tags:
            href = a.get("href")
            if not href:
                continue
            full_u = urljoin(url, href)
            if urlparse(full_u).netloc == base_domain and full_u not in visited_urls:
                visited_urls.add(full_u)
                internal_urls.append(full_u)

        for link, code in head_cached_parallel(internal_urls).items():
            if code >= 400:
                internal_errors.append({"url": link, "status": code})

        result["internal_link_errors"] = internal_errors

    except Exception as e:
        result["error"] = str(e)

    return result


# helpers.py  (add near the bottom)

def extract_page_issues(report: dict) -> list[str]:
    """
    Returns a list of plain‑English issue labels detected in a single-page
    `report` produced by full_seo_audit().
    """
    issues = []

    # ---- Meta data ----
    if report.get("title", {}).get("text") == "Missing":
        issues.append("missing <title>")
    if report.get("description", {}).get("text") == "Missing":
        issues.append("missing meta description")
    if report.get("duplicate_title"):
        issues.append("duplicate <title>")
    if report.get("duplicate_meta_description"):
        issues.append("duplicate meta description")

    # ---- Content / headings ----
    if report.get("duplicate_content"):
        issues.append("duplicate body content")
    if report.get("H1_content", "") == "":
        issues.append("no H1")
    if report.get("headings", {}).get("H1", 0) > 1:
        issues.append("multiple H1s")

    # ---- Links & images ----
    if report.get("empty_anchor_text_links", 0):
        issues.append("links with empty anchor text")
    if report.get("external_broken_links"):
        issues.append(f"{len(report['external_broken_links'])} broken outbound link(s)")
    if report.get("images", {}).get("images_without_alt", 0):
        issues.append(f"{report['images']['images_without_alt']} image(s) without alt")

    # ---- Ratios & misc ----
    if report.get("word_stats", {}).get("anchor_ratio_percent", 0) > 15:
        issues.append("anchor‑text ratio > 15 %")
    if report.get("text_to_html_ratio_percent", 100) < 10:
        issues.append("text‑to‑HTML ratio < 10 %")
    if not report.get("schema", {}).get("json_ld_found"):
        issues.append("no JSON‑LD schema")

    return issues


def ai_analysis(report, page_issues_map=None):

    issues_block = "\n".join(
        f"- **{url}**: " + ", ".join(problems)
        for url, problems in page_issues_map.items()
    ) or "✅ No page‑level issues detected."

    prompt = f"""You are an advanced SEO and web performance analyst. I am providing a JSON-formatted audit report of a website. This JSON includes data for individual URLs covering:
- HTTP/HTTPS status and response codes (including 4xx and 5xx errors)
- Page speed and response time
- Metadata (title, description, length, duplication)
- Content elements (word count, heading structure, text-to-HTML ratio)
- Link data (internal/external links, anchor text quality, redirects)
- Image data (alt tag presence, broken images)
- Schema markup presence
- Indexing and crawling restrictions (robots.txt, meta robots)

Your response should follow this structure:

### 🧠 AI-Powered SEO Summary

Then provide a detailed analysis, structured into these sections:

1. **Overall Health Summary**
   Brief summary of the site's technical SEO status.

2. **Strengths**
   Highlight technical strengths (e.g. HTTPS, schema usage, fast load times).

3. **Issues to Fix**
    Include only issues that are detected in the audit report.

4. **Critical Page-Level Errors**
   List problematic URLs and their specific technical issues.

5. **Actionable Recommendations**
   Give clear steps to improve technical SEO, indexing, crawlability, and UX.

---

Important:
- Parse the full report without skipping fields.
- Do NOT return your output as JSON.
- Do NOT include triple backticks or code blocks.
- Make the response client-friendly, as if it’s going into a formal audit report.
- Maintain clean structure, use bullet points and sections for clarity.

issues_block: {issues_block}
[SEO_REPORT]: {report}
"""


    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Error during Gemini API call: {e}\n\nDetails: {response.text}"

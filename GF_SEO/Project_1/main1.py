import time
from datetime import datetime
from urllib.parse import urlparse, urljoin, urlunparse

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import io
import markdown2
import undetected_chromedriver as uc
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import (
    ai_analysis,
    display_wrapped_json,
    full_seo_audit,
    get_rendered_html,
    should_skip_url,
    extract_page_issues,
    get_urls_from_sitemap,
)

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Remove trailing slashes + query params so the same page isn’t re‑crawled."""
    parsed = urlparse(url)
    clean_path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", "", ""))


def is_valid_link(href: str) -> bool:
    return href and not href.startswith("#") and not href.lower().startswith("javascript")


# ───────────── PDF / Markdown helpers ─────────────

def build_html_summary(summary_html: str, site_url: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"UTF-8\">
      <style>
        body {{ font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.6; }}
        h1   {{ font-size: 20pt; text-align: center; }}
        h2   {{ font-size: 16pt; margin-top: 20px; }}
        ul   {{ padding-left: 20px; }}
        li   {{ margin-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
        table, th, td {{ border: 1px solid #888; padding: 8px; word-wrap: break-word; white-space: normal; vertical-align: top; }}
        th {{ background-color: #f2f2f2; }}
      </style>
    </head>
    <body>
      <h1>AI SEO Summary Report</h1>
      <p><strong>Website:</strong> {site_url}</p>
      <p><strong>Date:</strong> {date_str}</p>
      {summary_html}
    </body>
    </html>
    """


def markdown_to_html(md: str) -> str:
    return markdown2.markdown(md, extras=["tables", "fenced-code-blocks"])


def convert_to_pdf(html: str) -> bytes:
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()


# ───────────── Metric aggregator ─────────────

def compute_sitewide_metrics(seo_data):
    """Return a DataFrame with **Metric**, **Count**, **Pages** columns.

    *Count*  – number of pages that triggered the metric.
    *Pages*  – newline‑separated list of the affected page URLs.
    """
    from collections import defaultdict

    # use two maps so we can populate count + affected urls in a single pass
    counts: dict[str, int]           = defaultdict(int)
    affected: dict[str, list[str]]   = defaultdict(list)

    for page in seo_data:
        url  = page.get("url")
        rpt  = page.get("report", {})

        # ── Meta data ──────────────────────────────────────────
        if rpt.get("title", {}).get("text") == "Missing":
            counts["Missing Title Tags"] += 1
            affected["Missing Title Tags"].append(url)
        if rpt.get("description", {}).get("text") == "Missing":
            counts["Missing Meta Descriptions"] += 1
            affected["Missing Meta Descriptions"].append(url)
        if rpt.get("duplicate_title"):
            counts["Duplicate Title Tags"] += 1
            affected["Duplicate Title Tags"].append(url)
        if rpt.get("duplicate_meta_description"):
            counts["Duplicate Meta Descriptions"] += 1
            affected["Duplicate Meta Descriptions"].append(url)
        if rpt.get("duplicate_content"):
            counts["Duplicate Content"] += 1
            affected["Duplicate Content"].append(url)

        # ── Headings ───────────────────────────────────────────
        if rpt.get("H1_content", "") == "":
            counts["H1 Content Missing"] += 1
            affected["H1 Content Missing"].append(url)
        if rpt.get("headings", {}).get("H1", 0) > 1:
            counts["Excessive H1 Elements"] += 1
            affected["Excessive H1 Elements"].append(url)

        # ── Images ─────────────────────────────────────────────
        if (n := rpt.get("images", {}).get("images_without_alt", 0)):
            counts["Images Without Alt Attributes"] += n  # count images, not pages
            affected["Images Without Alt Attributes"].append(url)

        # ── Anchor text ───────────────────────────────────────
        if rpt.get("empty_anchor_text_links", 0):
            counts["Empty Anchor Text Links"] += 1
            affected["Empty Anchor Text Links"].append(url)
        if rpt.get("word_stats", {}).get("anchor_ratio_percent", 0) > 15:
            counts["High Anchor Word Ratio (%)"] += 1
            affected["High Anchor Word Ratio (%)"].append(url)

        # ── Schema / ratios ───────────────────────────────────
        if not rpt.get("schema", {}).get("json_ld_found", False):
            counts["JSON-LD Schema Absent"] += 1
            affected["JSON-LD Schema Absent"].append(url)
        if rpt.get("text_to_html_ratio_percent", 100) < 10:
            counts["Low Text-to-HTML Ratio (%)"] += 1
            affected["Low Text-to-HTML Ratio (%)"].append(url)

        # ── Broken outbound links ─────────────────────────────
        if rpt.get("external_broken_links"):
            counts["Pages With Broken Outbound Links"] += 1
            affected["Pages With Broken Outbound Links"].append(url)
            counts["Total Broken Outbound Links"] += len(rpt["external_broken_links"])
            # we purposely don’t list every broken link row – only page level

    rows = [
        {
            "Metric": m,
            "Count": counts[m],
            "Pages": "\n".join(affected[m])
        }
        for m in counts
    ]
    return pd.DataFrame(rows, columns=["Metric", "Count", "Pages"])

def compute_broken_link_summary(seo_data):
    rows = []

    for page in seo_data:
        url = page.get("url")
        rpt = page.get("report", {})
        broken_links = rpt.get("external_broken_links", [])

        if broken_links:
            broken_urls = [bl.get("url", "N/A") for bl in broken_links]
            rows.append({
                "Page URL": url,
                "Broken Link Count": len(broken_urls),
                "Broken Links": ", ".join(broken_urls)
            })

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Crawling fallback (old behaviour)
# ────────────────────────────────────────────────────────────────────────────────

def crawl_entire_site(start_url, max_pages=None):
    """Legacy crawler when sitemap is not used."""
    import undetected_chromedriver as uc
    from selenium.common.exceptions import TimeoutException

    visited = set()
    queue   = [start_url]
    reports = []
    total   = 1

    bar   = st.progress(0.0)
    status = st.empty()

    titles_seen, descs_seen, hashes_seen = set(), set(), set()

    def safe_get(driver, url, retries=2, wait=1.5):
        for _ in range(retries):
            try:
                driver.get(url)
                time.sleep(wait)
                return driver.page_source
            except Exception:
                time.sleep(2)
        return None

    opts = uc.ChromeOptions()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    driver = uc.Chrome(options=opts)
    driver.set_page_load_timeout(30)

    try:
        while queue:
            if max_pages and len(visited) >= max_pages:
                break
            idx = total - len(queue)
            url = queue.pop(0)
            if should_skip_url(url):
                continue
            html = safe_get(driver, url)
            if not html:
                reports.append({"url": url, "report": {"error": "render failed"}})
                continue
            soup = BeautifulSoup(html, "html.parser")
            visited.add(url)
            rpt = full_seo_audit(url, titles_seen, descs_seen, hashes_seen, html)
            reports.append({"url": url, "report": rpt})

            base_domain = urlparse(start_url).netloc
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not is_valid_link(href):
                    continue
                ful = urljoin(url, href)
                if urlparse(ful).netloc == base_domain and ful not in visited and ful not in queue:
                    queue.append(ful)
                    total += 1
            bar.progress((idx + 1) / total)
            status.text(f"Crawled {idx+1} of approx {total}: {url}")
    finally:
        driver.quit()
        bar.progress(1.0)
        status.text("✅ Crawl completed")
    return reports


# ────────────────────────────────────────────────────────────────────────────────
# Streamlit application entry point
# ────────────────────────────────────────────────────────────────────────────────

def main():
    
    st.title("Full‑Site SEO Auditor")

    # ── UI controls ───────────────────────────────────────────────
    start_url   = st.text_input("Enter the homepage URL (e.g., https://example.com)")
    use_sitemap = st.checkbox("🔍 Use sitemap.xml for page discovery (faster, recommended)")
    st.caption("This will crawl all internal pages and analyze them.")
    limit_pages = st.checkbox("✅ Limit crawl to 200 pages max?")

    # ── Start button ──────────────────────────────────────────────
    if st.button("Start Full Site Audit"):
        if not start_url:
            st.warning("Please enter a valid URL.")
            st.stop()

        if not start_url.startswith(("http://", "https://")):
            start_url = "https://" + start_url.strip()

        parsed = urlparse(start_url)
        if parsed.netloc == "gauravguptastudio.com":
            start_url = "https://www.gauravguptastudio.com"

        # ── 1) Gather URLs ────────────────────────────────────────
        with st.spinner("Gathering links..."):
            if use_sitemap:
                sitemap_url   = start_url.rstrip("/") + "/sitemap.xml"
                urls_to_audit = get_urls_from_sitemap(sitemap_url)
                if not urls_to_audit:
                    st.error("❌ No URLs found in sitemap or sitemap is inaccessible.")
                    return
            else:
                urls_to_audit = [start_url]

            st.session_state["urls_to_audit"] = urls_to_audit

    # ── Preview + Trigger Audit button ────────────────────────────
    if "urls_to_audit" in st.session_state and "seo_data" not in st.session_state:
        urls_to_audit = st.session_state["urls_to_audit"]
        st.markdown("### 🔗 Preview of Pages to Audit")
        st.write(f"**Total pages found:** {len(urls_to_audit)}")
        with st.expander("Click to view URLs"):
            st.dataframe(pd.DataFrame(urls_to_audit, columns=["URL"]))
        if st.button("🚀 Start Audit"):
            # ── 2) Audit pages ────────────────────────────────────
            with st.spinner("Auditing site pages..."):
                titles_seen, descs_seen, hashes_seen = set(), set(), set()
                all_reports = []

                if use_sitemap:
                    bar   = st.progress(0.0)
                    stat  = st.empty()
                    eta   = st.empty()

                    start_time = time.time()
                    total      = len(urls_to_audit)

                    # single persistent driver
                    options = uc.ChromeOptions()
                    options.add_argument("--headless")
                    options.add_argument("--no-sandbox")
                    driver = uc.Chrome(options=options)
                    driver.set_page_load_timeout(15)

                    def audit_one(u):
                        html = get_rendered_html(u, driver)
                        return {"url": u, "report": full_seo_audit(u, titles_seen, descs_seen, hashes_seen, html)}

                    try:
                        with ThreadPoolExecutor(max_workers=6) as pool:
                            futures = {pool.submit(audit_one, u): u for u in urls_to_audit}
                            for i, fut in enumerate(as_completed(futures)):
                                res = fut.result()
                                all_reports.append(res)

                                # progress UI
                                bar.progress((i + 1) / total)
                                stat.markdown(f"🔍 **Audited {i+1}/{total}:** [`{res['url']}`]({res['url']})")
                                elapsed   = time.time() - start_time
                                remaining = (elapsed / (i + 1)) * (total - i - 1)
                                mm, ss    = divmod(int(remaining), 60)
                                eta.markdown(f"⏳ Estimated time left: **{mm} m {ss} s**")

                        bar.progress(1.0)
                        stat.markdown("✅ All pages audited.")
                        eta.empty()
                    finally:
                        driver.quit()

                else:
                    all_reports = crawl_entire_site(start_url, max_pages=200 if limit_pages else None)

                # ── Save results to session ─────────────────────
                page_issues_map = {
                    pg["url"]: extract_page_issues(pg["report"])
                    for pg in all_reports
                }
                page_issues_map = {u: iss for u, iss in page_issues_map.items() if iss}

                st.session_state["seo_data"]        = all_reports
                st.session_state["page_issues_map"] = page_issues_map
                st.session_state["ai_summary"]      = None
                st.session_state["ai_summary_time"] = None

            st.success("✅ Audit complete!")

    # ── Report views ───────────────────────────────────────
    if "seo_data" in st.session_state:
        view = st.radio("Choose report view:", ["📈 Raw SEO Report", "🤖 AI SEO Summary"])

        if view == "📈 Raw SEO Report":
            display_wrapped_json(st.session_state["seo_data"])

        else:
            metrics_df = compute_sitewide_metrics(st.session_state["seo_data"])
            st.markdown("### 🔗 Broken Outbound Links by Page")

            broken_summary_df = compute_broken_link_summary(st.session_state["seo_data"])

            if not broken_summary_df.empty:
                st.dataframe(broken_summary_df)
                st.download_button(
                    "📥 Download Broken Links Summary",
                    broken_summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="broken_links_summary.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ No broken outbound links found.")

            st.markdown("### 📈 Full SEO Issue Metrics (Calculated from All Pages)")
            st.dataframe(metrics_df)

            def generate_summary():
                return ai_analysis(
                    st.session_state["seo_data"],
                    st.session_state["page_issues_map"]
                )

            if st.button("♻️ Regenerate AI Summary") or st.session_state.get("ai_summary") is None:
                with st.spinner("Generating AI summary..."):
                    st.session_state["ai_summary"] = generate_summary()
                    st.session_state["ai_summary_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")

            raw = st.session_state["ai_summary"]
            gen_t = st.session_state.get("ai_summary_time", "")
            html = build_html_summary(markdown_to_html(raw), start_url)

            st.markdown("### 🤖 AI SEO Summary Preview")
            if gen_t:
                st.caption(f"Last generated: {gen_t}")
            st.markdown(raw)

            st.download_button(
                "📅 Download SEO Summary as PDF",
                convert_to_pdf(html),
                "seo_summary.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()

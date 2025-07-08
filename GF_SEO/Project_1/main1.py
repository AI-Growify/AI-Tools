import streamlit as st
from helpers import ai_analysis, display_wrapped_json, full_seo_audit, get_rendered_html
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup
import requests
import time
import re
from datetime import datetime
from xhtml2pdf import pisa
import io
import markdown2
import pandas as pd
from collections import defaultdict

# --- Normalize and Clean URLs ---
def normalize_url(url):
    parsed = urlparse(url)
    clean_path = parsed.path.rstrip('/')
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, '', '', ''))

def is_valid_link(href):
    return (
        href and
        not href.startswith('#') and
        not href.lower().startswith('javascript')
    )

# --- Convert Markdown to Styled HTML PDF ---
def build_html_summary(summary_html: str, site_url: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.6; }}
            h1 {{ font-size: 20pt; text-align: center; }}
            h2 {{ font-size: 16pt; margin-top: 20px; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 10px; }}
            table {{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
            }}
            table, th, td {{
                border: 1px solid #888;
                padding: 8px;
                word-wrap: break-word;
                white-space: normal;
                vertical-align: top;
            }}
            th {{
                background-color: #f2f2f2;
            }}
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
    return html

def markdown_to_html(text):
    return markdown2.markdown(text, extras=["tables", "fenced-code-blocks"])

def convert_to_pdf(html: str) -> bytes:
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

# --- Metric Aggregator ---
def compute_sitewide_metrics(seo_data):
    metrics_count = defaultdict(int)

    for page in seo_data:
        report = page.get("report", {})

        # --- Basic Metadata Checks ---
        if report.get("title", {}).get("text", "") == "Missing":
            metrics_count["Missing Title Tags"] += 1
        if report.get("description", {}).get("text", "") == "Missing":
            metrics_count["Missing Meta Descriptions"] += 1
        if report.get("duplicate_title"):
            metrics_count["Duplicate Title Tags"] += 1
        if report.get("duplicate_meta_description"):
            metrics_count["Duplicate Meta Descriptions"] += 1
        if report.get("duplicate_content"):
            metrics_count["Duplicate Content"] += 1

        # --- Heading Structure ---
        if report.get("H1_content", "") == "":
            metrics_count["H1 Content Missing"] += 1
        if report.get("headings", {}).get("H1", 0) > 1:
            metrics_count["Excessive H1 Elements"] += 1

        # --- Image Checks ---
        if report.get("images", {}).get("images_without_alt", 0):
            metrics_count["Images Without Alt Attributes"] += report["images"]["images_without_alt"]

        # --- Anchor Text Checks ---
        if report.get("empty_anchor_text_links", 0):
            metrics_count["Empty Anchor Text Links"] += report["empty_anchor_text_links"]
        if report.get("word_stats", {}).get("anchor_ratio_percent", 0) > 15:
            metrics_count["High Anchor Word Ratio (%)"] += 1

        # --- Schema & HTML Ratio ---
        if not report.get("schema", {}).get("json_ld_found", False):
            metrics_count["JSON-LD Schema Absent"] += 1
        if report.get("text_to_html_ratio_percent", 100) < 10:
            metrics_count["Low Text-to-HTML Ratio (%)"] += 1

        # --- External Broken Links (NEW) ---
        external_links = report.get("external_broken_links", [])
        if external_links:
            metrics_count["Pages With Broken Outbound Links"] += 1
            metrics_count["Total Broken Outbound Links"] += len(external_links)

    return pd.DataFrame(metrics_count.items(), columns=["Metric", "Count"])


# --- Crawler Function ---
def crawl_entire_site(start_url):
    visited = set()
    queue = [start_url]
    all_reports = []
    total_to_crawl = 1
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Add duplication trackers here
    titles_seen = set()
    descs_seen = set()
    content_hashes_seen = set()

    while queue:
        current_index = total_to_crawl - len(queue)
        current_url = queue.pop(0)
        normalized_current = normalize_url(current_url)

        if normalized_current in visited:
            continue

        status_text.text(f"🔍 Auditing {current_url} ({current_index + 1} of approx. {total_to_crawl})")

        try:
            html = get_rendered_html(current_url)
            if not html:
                all_reports.append({"url": current_url, "report": {"error": f"Could not render page: {current_url}"}})
                continue

            soup = BeautifulSoup(html, "html.parser")
            visited.add(normalized_current)
            report = full_seo_audit(current_url, titles_seen, descs_seen, content_hashes_seen, html)
            all_reports.append({"url": current_url, "report": report})

            base = urlparse(start_url).netloc
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not is_valid_link(href):
                    continue
                full_url = urljoin(current_url, href)
                normalized_url = normalize_url(full_url)
                if urlparse(normalized_url).netloc == base and normalized_url not in visited and normalized_url not in queue:
                    queue.append(normalized_url)
                    total_to_crawl += 1

            progress_bar.progress((current_index + 1) / total_to_crawl)
            time.sleep(0.5)

        except Exception as e:
            all_reports.append({"url": current_url, "error": str(e)})

    status_text.text("✅ Crawl completed!")
    progress_bar.progress(1.0)
    return all_reports

# --- Streamlit App ---
def main():
    st.title("Full-Site SEO Auditor")
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        **Welcome to the Full-Site SEO Auditor!**  
        This tool crawls your entire website and analyzes all internal pages for common SEO issues.  

        **📌 How it works:**
        - Enter your website's **homepage URL** in the input box (e.g., `https://example.com`).
        - Click **"Start Full Site Audit"**.
        - The tool will visit all internal links, render the pages, and perform a detailed SEO analysis.

        **⚡ Features of the Tool:**
        - Finds missing or duplicate **title tags** and **meta descriptions**.
        - Checks for **H1** structure issues.
        - Identifies **images missing alt attributes**.
        - Detects **empty anchor text links**.
        - Measures **text-to-HTML ratio** (important for content-heavy pages).
        - Checks for **broken outbound links**.
        - Verifies if **JSON-LD schema** is present.

        **✅ After Crawling:**
        - View the **Raw SEO Report** with page-by-page JSON output.
        - See a **Full SEO Issue Metrics** summary table.
        - Generate an **AI-powered SEO Summary** for executive reporting.
        - **Download** the AI summary as a PDF with all key insights.

        **💡 Tips:**
        - Make sure your site is publicly accessible (no login walls or blocks).
        - For best results, ensure links are well-structured (relative or absolute).
        - Large sites may take longer to crawl – please be patient!

        Enjoy optimizing! 🚀
        """)

    start_url = st.text_input("Enter the homepage URL (e.g., https://example.com)")
    st.caption("This will crawl all internal pages and analyze them.")

    if st.button("Start Full Site Audit"):
        if not start_url:
            st.warning("Please enter a valid URL.")
            return
        if not start_url.startswith("http://") and not start_url.startswith("https://"):
            start_url = "https://" + start_url.strip()

        with st.spinner("Crawling and analyzing site..."):
            full_report = crawl_entire_site(start_url)
            st.session_state["seo_data"] = full_report
            st.session_state["ai_summary"] = None
            st.session_state["ai_summary_time"] = None

        st.success("✅ Crawl complete!")

    if "seo_data" in st.session_state:
        view = st.radio("Choose report view:", ["📊 Raw SEO Report", "🤖 AI SEO Summary"])

        if view == "📊 Raw SEO Report":
            display_wrapped_json(st.session_state["seo_data"])

        elif view == "🤖 AI SEO Summary":
            metrics_df = compute_sitewide_metrics(st.session_state["seo_data"])
            st.markdown("### 📊 Full SEO Issue Metrics (Calculated from All Pages)")
            st.dataframe(metrics_df)

            if st.button("♻️ Regenerate AI Summary"):
                with st.spinner("Regenerating..."):
                    st.session_state["ai_summary"] = ai_analysis(st.session_state["seo_data"])
                    st.session_state["ai_summary_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
            elif "ai_summary" not in st.session_state or st.session_state["ai_summary"] is None:
                with st.spinner("Generating summary..."):
                    st.session_state["ai_summary"] = ai_analysis(st.session_state["seo_data"])
                    st.session_state["ai_summary_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")

            raw_summary = st.session_state["ai_summary"]
            generated_time = st.session_state.get("ai_summary_time", "")
            html_friendly = markdown_to_html(raw_summary)
            html = build_html_summary(html_friendly, start_url)

            st.markdown("### 🧠 AI SEO Summary Preview")
            if generated_time:
                st.caption(f"Last generated: {generated_time}")
            st.markdown(raw_summary)

            pdf_bytes = convert_to_pdf(html)
            st.download_button(
                label="📥 Download SEO Summary as PDF",
                data=pdf_bytes,
                file_name="seo_summary.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()  
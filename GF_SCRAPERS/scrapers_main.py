import streamlit as st
import pandas as pd
import re
import time
import os
import requests
from apify_client import ApifyClient
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import tldextract

# --- API KEYS ---
API_TOKEN = os.getenv("apify_api")
ACTOR_ID = os.getenv("actor_id")
WHATCMS_API = os.getenv("whatcms_api")

# Constants for contact scraping
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'
}

SOCIAL_DOMAINS = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com', 'pinterest.com']
def main():
    # --- Utility Functions for Contact Extraction ---
    def get_full_url(base_url, link):
        return urljoin(base_url, link)

    def is_internal(base_url, link):
        parsed_base = urlparse(base_url)
        parsed_link = urlparse(link)
        return (parsed_link.netloc == "" or parsed_link.netloc == parsed_base.netloc)

    def extract_contact_details(html):
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        emails = list(set(re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)))
        phones = list(set(re.findall(r'\+?\d[\d\s\-().]{8,}', text)))
        social_links = []
        for a in soup.find_all('a', href=True):
            for domain in SOCIAL_DOMAINS:
                if domain in a['href']:
                    social_links.append(a['href'])
        return {
            'emails': emails,
            'phones': phones,
            'social_links': list(set(social_links))
        }

    def get_candidate_links(soup, base_url):
        contact_keywords = ['contact', 'about', 'support', 'team']
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if any(word in href for word in contact_keywords):
                full_url = get_full_url(base_url, a['href'])
                if is_internal(base_url, full_url):
                    links.add(full_url)
        return list(links)

    def scrape_website(base_url):
        all_emails, all_phones, all_socials = set(), set(), set()
        try:
            res = requests.get(base_url, headers=HEADERS, timeout=10)
            res.raise_for_status()
        except Exception as e:
            return None
        soup = BeautifulSoup(res.text, 'html.parser')
        data = extract_contact_details(res.text)
        all_emails.update(data['emails'])
        all_phones.update(data['phones'])
        all_socials.update(data['social_links'])
        internal_links = get_candidate_links(soup, base_url)
        for link in internal_links:
            try:
                sub_res = requests.get(link, headers=HEADERS, timeout=10)
                sub_res.raise_for_status()
                data = extract_contact_details(sub_res.text)
                all_emails.update(data['emails'])
                all_phones.update(data['phones'])
                all_socials.update(data['social_links'])
            except:
                continue
        return {
            'website': base_url,
            'emails': list(all_emails),
            'phones': list(all_phones),
            'social_links': list(all_socials)
        }

    # ------------------ Streamlit UI ------------------
   
    st.title("PitchKit: Designer Lead Extractor")
    with st.expander("How to Use This Tool"):
        st.markdown("""
        **Welcome to PitchKit: Designer Lead Extractor!**  
        This tool is designed for marketing teams, agencies, and brand managers who want to **quickly research and collect leads** for designer brands and boutiques.

        **What this tool can do:**
        - **Scrape contact details** (emails, phones, social links) from designer/boutique websites.
        - **Look up Instagram handles** to get follower counts, bio, verification, and business status.
        - **Calculate Instagram engagement rates** for any public profile.
        - **Search designer brand websites** and **detect their CMS** (Shopify, WooCommerce, etc.).

        **✅ How to use each section:**

        **1️⃣ Contact Scraper**
        - Paste one or more website URLs (one per line).
        - Click **"Scrape Contact Info"**.
        - Get a table of emails, phones, social links with a CSV download.

        **2️⃣ Instagram Handle Lookup**
        - Enter Instagram handles (comma-separated).
        - Click **"Fetch Instagram Info"**.
        - See bios, followers, business status, etc. in a table.

        **3️⃣ Engagement Rate Calculator**
        - Enter one Instagram handle.
        - Click **"Calculate Engagement Rate"**.
        - View the estimated engagement % from a live web calculator.

        **4️⃣ Designer Search + CMS Detection**
        - Enter designer brand names (one per line).
        - Click **"Search Designers"**.
        - Tool will find top official website links for each brand.
        - Select a link to see its **CMS platform** (e.g., Shopify).
        - Click **"Scrape Contact"** to get emails, phones, social links from that site.

        **Tips:**
        - Use official brand sites for best contact accuracy.
        - For Instagram lookups, ensure profiles are public.
        - Engagement rate calculator uses a live web scraper — be patient!
        - CMS detection helps you identify e-commerce platforms for partnership or competitor research.

        Ready to grow your lead pipeline? 
        """)

    # Input website URLs
    input_urls = st.text_area("Enter Website URLs (one per line)", height=200)
    urls = [url.strip() for url in input_urls.split("\n") if url.strip()]

    # Process URLs
    if st.button("Scrape Contact Info") and urls:
        scraped_results = []
        with st.spinner("Scraping websites..."):
            for url in urls:
                if not url.startswith("http"):
                    url = "https://" + url
                result = scrape_website(url)
                if result:
                    scraped_results.append(result)

        if scraped_results:
            df = pd.DataFrame(scraped_results)
            st.subheader("Extracted Contact Information")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="contact_details.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data was extracted from the provided URLs.")

    # ------------------ Instagram & Designer Info Section ------------------
    st.header("PitchKit Insights – Instagram & Designer Intelligence")
    st.markdown(
    "<p style='font-size: 20px; color: gray;'>Uncover designer trends, social handles, and Instagram insights – all in one place.</p>",
    unsafe_allow_html=True
)

    # Initialize session state
    for key in ["insta_usernames", "username", "designer_names", "insta_info", "engagement_rate"]:
        if key not in st.session_state:
            st.session_state[key] = ""

    if "search_results" not in st.session_state:
        st.session_state.search_results = {}

    if "cms_results" not in st.session_state:
        st.session_state.cms_results = {}

    if "selected_links" not in st.session_state:
        st.session_state.selected_links = {}

    # Instagram Info Section
    st.markdown("### Instagram Handle Lookup")
    st.session_state["insta_usernames"] = st.text_area(
        "Enter Instagram Handles (comma separated)",
        value=st.session_state["insta_usernames"]
    )

    if st.button("Fetch Instagram Info"):
        usernames = [u.strip().lstrip("@") for u in st.session_state["insta_usernames"].split(",") if u.strip()]
        if usernames:
            client = ApifyClient(API_TOKEN)
            run = client.actor(ACTOR_ID).call(run_input={"usernames": usernames})
            dataset_id = run["defaultDatasetId"]
            info = {}
            for item in client.dataset(dataset_id).iterate_items():
                u = item.get("username")
                if u:
                    info[u] = item
            df = pd.DataFrame([{ 
                "Username": u,
                "Full Name": info[u].get("fullName"),
                "Followers": info[u].get("followersCount"),
                "Following": info[u].get("followsCount"),
                "Verified": info[u].get("verified"),
                "Business": info[u].get("isBusinessAccount"),
                "Joined Recently": info[u].get("joinedRecently"),
                "URL": info[u].get("url"),
                "Bio": info[u].get("biography")
            } for u in usernames if u in info])
            st.session_state["insta_info"] = df

    if isinstance(st.session_state["insta_info"], pd.DataFrame):
        st.dataframe(st.session_state["insta_info"])

    # Engagement Rate Section
    st.markdown("### Engagement Rate Calculator")
    st.session_state["username"] = st.text_input(
        "Enter Instagram Handle (single)",
        value=st.session_state["username"]
    )

    def get_engagement_rate(username):
        driver = create_driver()
        try:
            driver.get("https://www.clickanalytic.com/free-instagram-engagement-calculator/")
            input_box = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            input_box.clear()
            input_box.send_keys(username)
            submit_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Check Engagement Rate')]"))
            )
            submit_btn.click()
            WebDriverWait(driver, 20).until(lambda d: "%" in d.page_source)
            time.sleep(3)
            result_blocks = driver.find_elements(By.XPATH, "//div[contains(@class,'et_pb_text_inner')]")
            for block in result_blocks:
                if "%" in block.text:
                    match = re.search(r"\d+\.?\d*%", block.text)
                    if match:
                        rate = match.group(0)
                        driver.quit()
                        return rate
            driver.quit()
            return "Engagement rate not found."
        except Exception as e:
            driver.quit()
            return f"Error: {e}"

    def create_driver():
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        driver = webdriver.Chrome(options=options)
        stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        return driver

    if st.button("Calculate Engagement Rate"):
        clean_user = st.session_state["username"].strip().lstrip("@")
        if clean_user:
            st.session_state["engagement_rate"] = get_engagement_rate(clean_user)

    if st.session_state["engagement_rate"]:
        st.success(f"Engagement Rate: {st.session_state['engagement_rate']}")

    # Designer Search + CMS Detection
    st.markdown("### Designer Search Links + CMS Detection")
    st.session_state["designer_names"] = st.text_area(
        "Enter Designer Names (one per line)",
        value=st.session_state["designer_names"]
    )

    from selenium.webdriver.common.by import By

    def fetch_top_links(query):
        try:
            driver = create_driver()
            driver.get("https://html.duckduckgo.com/html/")
            search_bar = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "search_form_input_homepage"))
            )
            search_bar.clear()
            search_bar.send_keys(query)
            driver.find_element(By.ID, "search_button_homepage").click()
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.web-result"))
            )
            result_blocks = driver.find_elements(By.CSS_SELECTOR, "div.web-result")
            top_links = []
            for block in result_blocks:
                anchor = block.find_element(By.CSS_SELECTOR, "a.result__url")
                href = anchor.get_attribute("href")
                if href and not any(domain in href for domain in ["instagram.com", "twitter.com", "x.com", "youtube.com", "youtu.be"]):
                    top_links.append(href)
                if len(top_links) == 3:
                    break
            driver.quit()
            return top_links
        except Exception as e:
            driver.quit()
            return [f"Error: {e}"]

    def detect_cms(url):
        try:
            response = requests.get(
                "https://whatcms.org/APIEndpoint/Detect",
                params={"key": WHATCMS_API, "url": url},
                timeout=10
            )
            data = response.json()
            if data.get("result", {}).get("code") == 200:
                cms = data["result"].get("name", "Unknown")
                confidence = data["result"].get("confidence", "N/A")
                return cms, confidence
            else:
                return "Unknown", "N/A"
        except Exception as e:
            return f"Error: {e}", "N/A"

    if st.button("Search Designers"):
        names = [n.strip() for n in st.session_state["designer_names"].split("\n") if n.strip()]
        for name in names:
            links = fetch_top_links(f"{name} designer official website")
            st.session_state.search_results[name] = links

    if "search_results" in st.session_state:
        for name, links in st.session_state.search_results.items():
            if links:
                st.subheader(f"{name}")
                selected_link = st.radio(
                    f"Top links for {name}", links,
                    key=f"radio_{name}",
                    index=0
                )
                st.session_state.selected_links[name] = selected_link
                if (name, selected_link) not in st.session_state.cms_results:
                    cms, confidence = detect_cms(selected_link)
                    st.session_state.cms_results[(name, selected_link)] = (cms, confidence)
                cms, confidence = st.session_state.cms_results.get((name, selected_link), ("Unknown", "N/A"))
                st.write(f"*Selected Link:* [{selected_link}]({selected_link})")
                st.write(f"*CMS:* {cms} | *Confidence:* {confidence}")
                if st.button(f"Scrape Contact for {name}", key=f"scrape_{name}"):
                    data = scrape_website(selected_link)
                    if data:
                        st.write("Emails:", data['emails'])
                        st.write("Phones:", data['phones'])
                        st.write("Socials:", data['social_links'])
            else:
                st.warning(f"No valid links found for {name}.")

if __name__ == "__main__":
    main()
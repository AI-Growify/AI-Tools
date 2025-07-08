import streamlit as st
import json
import time
import requests
from bs4 import BeautifulSoup
import re
import yake
import google.generativeai as genai
import random
import os
from dotenv import load_dotenv  # ✅ Added for .env support

def main():
    # ---------- CONFIG ----------
    st.title("PROMPTLY - The AI Blog Idea Generator")
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        **Welcome to PROMPTLY - The AI Blog Idea Generator!**  
        This tool is designed for content strategists and marketing teams to instantly brainstorm **fresh, engaging blog ideas** tailored to your brand.

        **📌 How it works:**
        - Select your **brand** from the list. All saved details (description, website, example blogs) will load automatically.
        - Optionally add **extra notes** about campaigns, seasonal themes, or target audiences.
        - Click **"Generate Blog Ideas"** to instantly get **5 unique, high-quality blog titles** designed to match premium fashion/lifestyle content standards.

        **⚡ What the tool does:**
        - Crawls the brand’s website to extract **internal product/category pages**.
        - Analyzes site content to extract **relevant, non-branded keywords**.
        - Uses advanced AI to craft **original, non-repetitive blog titles** suitable for high-end audiences.
        - Ensures topics encourage **natural internal linking** (e.g., "explore our latest collection") without sounding forced or generic.

        **✅ Features:**
        - Always generates **different and creative** ideas on each run.
        - Designed for **high-end fashion, lifestyle, and luxury brands**.
        - Keeps suggestions **non-branded** to expand SEO reach and customer interest.

        **💡 Tips:**
        - Add extra notes to fine-tune ideas for campaigns or seasons.
        - Review the extracted keywords for insight into your site's current content themes.
        - Use the generated titles as inspiration for blogs, ad copies, or social posts.

        Ready to brainstorm like a pro? 🚀
        """)

    # ---------- LOAD BRAND DATA ----------
    data_path = os.path.join(os.path.dirname(__file__), "brand_data.json")
    with open(data_path, "r") as f:
        brand_data = json.load(f)
    brand_dict = {brand["brand_name"]: brand for brand in brand_data}
    brand_names = list(brand_dict.keys())

    # ---------- SELECT BRAND ----------
    selected_brand = st.selectbox("🔍 Select a Brand", brand_names)
    brand_info = brand_dict[selected_brand]
    brand_desc = brand_info.get("description", "")
    brand_example_blogs = brand_info.get("example_blogs", "")

    # ---------- ADDITIONAL NOTES ----------
    user_notes = st.text_area("📝 Additional Brand Notes (optional)", "")

    st.subheader("📌 Brand Overview")
    with st.expander("Show Brand Info"):
        for key, value in brand_info.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

    # ---------- WEBSITE CRAWLING ----------
    def get_internal_links(base_url, max_pages=5, max_time=20):
        visited = set()
        to_visit = [base_url]
        internal_links = []
        allowed_keywords = ["product", "shop", "about", "collection", "category", "wedding", "jewellery", "ethnic", "clothing"]
        disallowed_keywords = ["cart", "account", "login", "wishlist", "checkout", "signup", "register"]

        start_time = time.time()

        while to_visit and len(internal_links) < max_pages:
            if time.time() - start_time > max_time:
                break

            random.shuffle(to_visit)
            url = to_visit.pop(0)

            if url in visited or any(bad in url for bad in disallowed_keywords):
                continue

            visited.add(url)
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")

                if any(word in url for word in allowed_keywords):
                    internal_links.append(url)

                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if href.startswith("/"):
                        href = base_url.rstrip("/") + href
                    if href.startswith(base_url) and href not in visited:
                        to_visit.append(href)
            except:
                continue

        return internal_links

    def extract_text_from_urls(urls):
        text = ""
        for url in urls:
            try:
                r = requests.get(url, timeout=5)
                soup = BeautifulSoup(r.text, "html.parser")
                for script in soup(["script", "style"]): script.extract()
                text += " " + soup.get_text(separator=' ')
            except:
                continue
        return re.sub(r'\s+', ' ', text)

    def get_keywords(text, n=15):
        kw_extractor = yake.KeywordExtractor(top=n, stopwords=None)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    # ---------- CACHED PIPELINE ----------
    @st.cache_data(ttl=3600)
    def cached_crawl_and_keywords(website_url):
        urls = get_internal_links(website_url)
        text_data = extract_text_from_urls(urls)
        keywords = get_keywords(text_data)
        return urls, text_data, keywords

    # ---------- GEMINI SETUP ----------
    load_dotenv()  # ✅ Load .env from root or current folder
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    if not GEMINI_API_KEY:
        st.error("❌ Gemini API key not found. Please add it to your `.env` file as `GEMINI_API_KEY=...`")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    # ---------- GENERATE BLOG IDEAS ----------
    if st.button("🧠 Generate Blog Ideas"):
        with st.spinner("Crawling website and crafting unique blog ideas..."):
            website_url = brand_info.get("website", "")
            if not website_url:
                st.error("❌ Website URL not found.")
            else:
                urls, text_data, keywords = cached_crawl_and_keywords(website_url)

                context_snippet = f"""
Brand Description:
{brand_desc[:500]}

Example Blogs:
{brand_example_blogs}

Additional Notes:
{user_notes}

Extracted Keywords from website: {", ".join(keywords)}
"""

                blog_idea_prompt = f"""
You are a senior content strategist helping a premium brand come up with **original and creative blog ideas**.
Make sure that for each and everytime I run you the ideas are different and unique. It should not be repeated 
and the topic should be fresh and engaging. Think of something like femina or vogue or high-end fashion blogs, wherein 
the content is not generic and is very engaging and they smartly link to the products on the website. I want exactly that kind of content.
{context_snippet}

Now, based on the above, generate **5 fresh, unique, and engaging blog titles** that:
- Use non-branded keywords only (no brand name)
- Reflect real customer intent (how-to guides, styling tips, emotional connections, etc.)
- Are NOT generic or repeated — avoid clichés
- Allow for natural internal product linking on a website (e.g., "explore our latest collection")
- Match the tone and style of high-end fashion or lifestyle blogs
- Don’t make it robotic in any case — avoid using words like tapestry and words like that. Make it very engaging and human-like.
Ensure every title is blog-ready — catchy, crisp, and impactful.
"""

                try:
                    response = model.generate_content(blog_idea_prompt)
                    blog_ideas = response.text.strip()
                except Exception as e:
                    blog_ideas = f"❌ Error generating blog ideas: {e}"

                st.subheader("🎯 Blog Ideas")
                st.markdown(blog_ideas)

# Optional standalone runner
if __name__ == "__main__":
    main()

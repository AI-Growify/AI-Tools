FROM python:3.10-slim

WORKDIR /app

# ✅ Install OS dependencies (minimum stable setup)
RUN apt-get update && apt-get install -y \
    xvfb xauth libnss3 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 \
    wget unzip gnupg lsb-release xdg-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Install Chrome separately (stable way)
RUN wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y /tmp/chrome.deb && rm /tmp/chrome.deb

# ✅ Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

# ✅ Copy app code
COPY . .

# ✅ Environment setup
ENV CHROME_BIN=/usr/bin/google-chrome
ENV STREAMLIT_PORT=8501

EXPOSE 8501

# ✅ Streamlit run inside X virtual framebuffer (for headless Chrome)
CMD ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "streamlit", "run", "main_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

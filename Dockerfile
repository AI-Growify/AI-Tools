FROM python:3.10-slim

WORKDIR /app

# ✅ Pre-install system dependencies
COPY apt.txt .
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive xargs -a apt.txt apt-get install -y --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ✅ Install Chrome manually
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb

# ✅ Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ✅ Copy app code
COPY . .

# ✅ Environment config
ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/google-chrome
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV STREAMLIT_PORT=8501

# ✅ Expose Streamlit port
EXPOSE 8501

# ✅ Final Run Command with XVFB
CMD ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "streamlit", "run", "main_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies + Chrome via Google’s official apt repo
RUN apt-get update && apt-get install -y \
    wget gnupg curl xdg-utils apt-transport-https ca-certificates lsb-release \
    xvfb xauth libnss3 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# ✅ Add Google’s official signing key and Chrome repo
RUN curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list

# ✅ Install Google Chrome latest stable with correct dependencies
RUN apt-get update && apt-get install -y google-chrome-stable && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

# ✅ Copy your app
COPY . .

# ✅ Environment setup
ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/google-chrome
ENV STREAMLIT_PORT=8501

# ✅ Expose Streamlit port
EXPOSE 8501

# ✅ Production CMD (auto-start Streamlit)
CMD ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "streamlit", "run", "main_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

# ✅ [Optional] Debugging Tip:
# If debugging, you can TEMPORARILY switch CMD to below to access shell via Render:
# CMD ["sleep", "1000"]

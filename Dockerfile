FROM python:3.10-slim

WORKDIR /app

# ✅ Install OS dependencies + Chrome
RUN apt-get update && apt-get install -y \
    wget gnupg curl xdg-utils apt-transport-https ca-certificates lsb-release \
    xvfb xauth libnss3 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# ✅ Add Google Chrome’s official repository
RUN curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list

RUN apt-get update && apt-get install -y google-chrome-stable && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

# ✅ Copy application code
COPY . .

# ✅ Setup Streamlit configuration to disable onboarding prompts completely
RUN mkdir -p /root/.streamlit && \
    echo "[server]" > /root/.streamlit/config.toml && \
    echo "enableCORS = false" >> /root/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> /root/.streamlit/config.toml && \
    echo "headless = true" >> /root/.streamlit/config.toml && \
    echo "port = 8501" >> /root/.streamlit/config.toml && \
    echo "address = '0.0.0.0'" >> /root/.streamlit/config.toml && \
    echo "[browser]" >> /root/.streamlit/config.toml && \
    echo "gatherUsageStats = false" >> /root/.streamlit/config.toml

# ✅ Environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/google-chrome

EXPOSE 8501

# ✅ Final CMD - clean lock file, launch Xvfb and Streamlit properly
CMD bash -c "rm -rf /tmp/.X99-lock && Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && streamlit run main_dashboard.py --server.port=8501 --server.address=0.0.0.0"

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget gnupg curl xdg-utils apt-transport-https ca-certificates lsb-release \
    xvfb xauth libnss3 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# ✅ Google Chrome repo
RUN curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list

RUN apt-get update && apt-get install -y google-chrome-stable && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/google-chrome
ENV STREAMLIT_PORT=8501

EXPOSE 8501

# ✅ Run Xvfb manually then Streamlit → ensures port binding works
CMD bash -c "Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && streamlit run main_dashboard.py --server.port=8501 --server.address=0.0.0.0"

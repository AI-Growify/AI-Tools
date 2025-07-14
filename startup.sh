#!/bin/bash
echo "🚀 Starting Growify AI on Render..."

export DISPLAY=:99
export CHROME_BIN=/usr/bin/chromium-browser
export CHROMEDRIVER_PATH=/usr/bin/chromedriver

echo "🖥️  Starting virtual display..."
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 2

exec streamlit run main_dashboard.py --server.port=$PORT --server.address=0.0.0.0

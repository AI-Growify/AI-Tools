#!/bin/bash

echo "🚀 Starting Growify AI deployment on Render..."

# Set up environment variables
export DISPLAY=:99
export CHROME_BIN=/usr/bin/chromium-browser
export CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Check if running on Render
if [[ "$RENDER" == "true" ]]; then
    echo "📦 Running on Render - installing Chrome dependencies..."
    
    # Install Chrome and dependencies
    apt-get update
    apt-get install -y \
        chromium-browser \
        chromium-chromedriver \
        xvfb \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
        libxss1 \
        libappindicator3-1 \
        libnss3 \
        lsb-release \
        xdg-utils
    
    # Start Xvfb for headless display
    echo "🖥️  Starting virtual display..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    
    # Wait for Xvfb to start
    sleep 2
    
    # Verify Chrome installation
    if command -v chromium-browser &> /dev/null; then
        echo "✅ Chrome installed successfully"
        chromium-browser --version
    else
        echo "❌ Chrome installation failed"
        exit 1
    fi
    
    # Verify ChromeDriver installation
    if command -v chromedriver &> /dev/null; then
        echo "✅ ChromeDriver installed successfully"
        chromedriver --version
    else
        echo "❌ ChromeDriver installation failed"
        exit 1
    fi
fi

# Start the Streamlit app
echo "🎯 Starting Streamlit application..."
streamlit run main_dashboard.py --server.port=$PORT --server.address=0.0.0.0
class OCR:
    class TESSERACT:
        pass
    class EASYOCR:
        pass

DEBUG_PRINT = True
MAX_GATHER_ITERATIONS_MULTIPLIER = 2
OCR_METHOD = OCR.EASYOCR


def ChromeConfig(debug = False):
    from selenium import webdriver
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    if not debug:
        options.add_argument('--headless')
    #options.add_argument('--disable-gpu')
    options.add_argument('--disable-notifications')
    # options.add_argument('--window-size=1920,1080')
    # options.add_argument(
    #    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    # options.add_argument('--proxy-server=http://username:password@proxy.example.com:8080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    # options.add_argument('--start-maximized')
    options.add_argument('--lang=en-US')
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-running-insecure-content')
    # options.add_argument('--disable-javascript')
    #options.add_argument('--disable-cache')
    options.add_argument('--disable-sync')
    # options.add_argument('--disable-background-networking')
    options.add_argument('--disable-background-timer-throttling')
    options.add_argument('--disable-backgrounding-occluded-windows')
    options.add_argument('--disable-breakpad')
    # options.add_argument('--disable-client-side-phishing-detection')
    options.add_argument('--disable-default-apps')
    options.add_argument('--disable-hang-monitor')
    options.add_argument('--disable-ipc-flooding-protection')
    options.add_argument('--disable-prompt-on-repost')
    #options.add_argument('--disable-renderer-accessibility')
    #options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-speech-api')
    options.add_argument('--disable-remote-fonts')
    options.add_argument('--disable-translate')
    options.add_argument('--disable-component-update')
    return options
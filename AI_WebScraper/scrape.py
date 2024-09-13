import time
import random
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from stem import Signal
from stem.control import Controller
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def get_tor_session():
    session = requests.session()
    session.proxies = {
        'http': 'socks5h://localhost:9050',
        'https': 'socks5h://localhost:9050'
    }
    return session

def renew_tor_ip():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate()
        controller.signal(Signal.NEWNYM)

def scrape_web(target):
    print("Preparing to scrape...")

    # Rotate user agents
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    
    # Use a mix of request methods
    methods = ['requests', 'selenium', 'tor']
    method = random.choice(methods)
    
    try:
        if method == 'requests':
            # Simple requests with rotating user agent
            response = requests.get(target, headers=headers)
            html = response.text
        elif method == 'selenium':
            # Selenium with proxy rotation
            chrome_driver_path = "./Driver/chromedriver.exe"
            options = Options()
            options.add_argument(f'user-agent={ua.random}')
            
            # Uncomment and set up proxy if you have a list of proxies
            # options.add_argument('--proxy-server=http://your-proxy-ip:port')
            
            driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)
            try:
                driver.get(target)
                time.sleep(random.uniform(1, 3))  # Random delay
                html = driver.page_source
            finally:
                driver.quit()
        else:
            # Tor network
            session = get_tor_session()
            renew_tor_ip()
            response = session.get(target, headers=headers)
            html = response.text
        
        print("Page scraped successfully")
        return html
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""

def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )
    return cleaned_content

def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[x: x + max_length] for x in range(0, len(dom_content), max_length)
    ]

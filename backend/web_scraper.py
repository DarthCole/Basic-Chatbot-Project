import time
import os
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import logging

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedWebScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.chrome_options = self._setup_chrome_options()
        
    def _setup_chrome_options(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        return chrome_options
    
    def scrape_with_selenium(self, url, wait_time=5):
        driver = None
        try:
            logger.info(f"Scraping with Selenium: {url}")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            driver.get(url)
            
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            self._scroll_page(driver)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            pdf_links = self._extract_pdf_links(driver, url)
            content = self._extract_clean_text(soup)
            
            return {
                "url": url,
                "content": content,
                "pdf_links": pdf_links,
                "method": "selenium",
                "success": True,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Selenium scraping error: {str(e)}")
            return {
                "url": url,
                "content": "",
                "pdf_links": [],
                "method": "selenium",
                "success": False,
                "error": str(e)
            }
        finally:
            if driver:
                driver.quit()
    
    def scrape_with_playwright(self, url):
        if not HAS_PLAYWRIGHT:
            logger.warning("Playwright not available, skipping Playwright scraping")
            return {
                "url": url,
                "content": "",
                "pdf_links": [],
                "method": "playwright",
                "success": False,
                "error": "Playwright not installed"
            }
        
        try:
            logger.info(f"Scraping with Playwright: {url}")
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.new_page()
                
                page.goto(url)
                page.wait_for_load_state("networkidle")
                self._scroll_playwright(page)
                
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                pdf_links = self._extract_pdf_links_from_soup(soup, url)
                text_content = self._extract_clean_text(soup)
                
                browser.close()
                
                return {
                    "url": url,
                    "content": text_content,
                    "pdf_links": pdf_links,
                    "method": "playwright",
                    "success": True,
                    "content_length": len(text_content)
                }
                
        except Exception as e:
            logger.error(f"Playwright scraping error: {str(e)}")
            return {
                "url": url,
                "content": "",
                "pdf_links": [],
                "method": "playwright",
                "success": False,
                "error": str(e)
            }
    
    def scrape_static(self, url):
        try:
            logger.info(f"Scraping static: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_links = self._extract_pdf_links_from_soup(soup, url)
            content = self._extract_clean_text(soup)
            
            return {
                "url": url,
                "content": content,
                "pdf_links": pdf_links,
                "method": "static",
                "success": True,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Static scraping error: {str(e)}")
            return {
                "url": url,
                "content": "",
                "pdf_links": [],
                "method": "static",
                "success": False,
                "error": str(e)
            }
    
    def _scroll_page(self, driver):
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def _scroll_playwright(self, page):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
    
    def _extract_pdf_links(self, driver, base_url):
        try:
            pdf_links = []
            links = driver.find_elements(By.TAG_NAME, "a")
            
            for link in links:
                href = link.get_attribute("href")
                if href and href.lower().endswith('.pdf'):
                    full_url = urljoin(base_url, href)
                    pdf_links.append({
                        "url": full_url,
                        "text": link.text[:100]
                    })
            
            return pdf_links[:10]
        except:
            return []
    
    def _extract_pdf_links_from_soup(self, soup, base_url):
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and href.lower().endswith('.pdf'):
                full_url = urljoin(base_url, href)
                pdf_links.append({
                    "url": full_url,
                    "text": link.text[:100]
                })
        return pdf_links[:10]
    
    def _extract_clean_text(self, soup):
        for element in soup(["script", "style", "nav", "footer", "header", 
                           "iframe", "aside", "form", "button"]):
            element.decompose()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:15000]
    
    def scrape(self, url, method="auto"):
        methods_to_try = []
        
        if method == "auto":
            methods_to_try = ["static", "selenium"]
            if HAS_PLAYWRIGHT:
                methods_to_try.append("playwright")
        else:
            methods_to_try = [method]
        
        for method_name in methods_to_try:
            if method_name == "static":
                result = self.scrape_static(url)
            elif method_name == "selenium":
                result = self.scrape_with_selenium(url)
            elif method_name == "playwright":
                result = self.scrape_with_playwright(url)
            
            if result["success"] and len(result["content"]) > 100:
                logger.info(f"Success with {method_name}: {len(result['content'])} chars")
                return result
        
        return {
            "url": url,
            "content": "",
            "pdf_links": [],
            "method": "failed",
            "success": False,
            "error": "All scraping methods failed"
        }
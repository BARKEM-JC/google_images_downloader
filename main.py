from lib import kill_all_selenium_instances
import atexit
from google_downloader import Search, SearchConfig
from constants import ChromeConfig

if __name__ == '__main__':
    kill_all_selenium_instances()
    atexit.register(kill_all_selenium_instances)
    #search = Search('eggs', SearchConfig(image_limit=100))
    #search.StartThreads()
    Search('test', SearchConfig(
        ignore_text=False, humans_only=False, image_limit=20, chrome_options=ChromeConfig(False), scraper_processes=3,
        related=False,)).StartThreads()
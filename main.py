import time

from lib import kill_all_selenium_instances
import atexit
from google_downloader import Search, SearchConfig
from constants import ChromeConfig

if __name__ == '__main__':
    kill_all_selenium_instances()
    atexit.register(kill_all_selenium_instances)
    start_time = time.time()
    search = Search('eggs', SearchConfig(
        ignore_text=False, humans_only=False, image_limit=150, chrome_options=ChromeConfig(False), scraper_processes=1,
        related=False))
    search.StartThreads()
    print("Execution Time: ", time.time() - start_time - 5)
    #Search('test', SearchConfig(
    #    ignore_text=False, humans_only=False, image_limit=20, chrome_options=ChromeConfig(False), scraper_processes=3,
    #    related=False,)).StartThreads()
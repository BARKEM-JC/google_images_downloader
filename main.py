import time

from lib import kill_all_selenium_instances
#import atexit
from google_downloader import Search, SearchConfig
from constants import ChromeConfig

debug_mode = True

if __name__ == '__main__':
    #kill_all_selenium_instances()
    #atexit.register(kill_all_selenium_instances)
    start_time = time.time()
    search = Search('eggs', SearchConfig(image_limit=50, chrome_options=ChromeConfig(debug_mode)))
    search.StartThreads()
    print("Execution Time: ", time.time() - start_time - 5)
import atexit
import random
import sys
import threading
from selenium import webdriver
import constants
import psutil
import time


def exit_handler():
    kill_all_selenium_instances()

def get_selenium_instance_pids():
    arr = []
    try:
        import psutil
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if (process.name() == 'chrome.exe' or
                    process.name() == 'chromedriver' or
                    process.name() == 'geckodriver'):
                arr.append(process.pid)
    except Exception as e:
        print(f"An error occurred while terminating Selenium instances: {e}")
    return arr


def kill_all_selenium_instances():
    try:
        import psutil
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if (process.name() == 'chrome.exe' or
                    process.name() == 'chromedriver' or
                    process.name() == 'geckodriver'):
                process.terminate()
                print(f"Selenium instance with PID {process.pid} terminated.")
    except Exception as e:
        print(f"An error occurred while terminating Selenium instances: {e}")


def kill_process(pid):
    import psutil
    try:
        process = psutil.Process(pid)
        process.terminate()  # Terminate the process
        print(f"Process with PID {pid} terminated.")
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}.")
    except Exception as e:
        print(f"An error occurred while terminating process with PID {pid}: {e}")


def random_example_url():
    import string
    import random
    valid_chars = string.ascii_letters + string.digits + "/_-."
    path_length = random.randint(5, 20)
    path = ''.join(random.choice(valid_chars) for _ in range(path_length))
    domains = ['example.com', 'example.org', 'example.net', 'example.co']
    domain = random.choice(domains)
    url = f"https://{domain}/{path}"

    return url


def random_stress_test_websites():
    sites = ['https://thispersondoesnotexist.com/',
             'https://www.youtube.com/',
             "https://www.google.com/search?tbm=isch&q=cats"
             "https://www.reddit.com/"
             "https://ytroulette.com/"
             ]
    return sites[random.randrange(0, len(sites) - 1, 1)]


#FUCKED
class Benchmark():
    def __init__(self):
        self.Lock = threading.Lock()
        self.CPU_USAGE = 0
        self.SELENIUM_INSTANCES_CPU_USAGE = []
        pass

    def benchmark_cpu(self):
        import psutil
        import time
        import multiprocessing

        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n - 1) + fibonacci(n - 2)

        cpu_count = multiprocessing.cpu_count

        initial_time = time.time()
        initial_cpu_time = psutil.cpu_times()
        fibonacci(30)
        final_time = time.time()
        final_cpu_time = psutil.cpu_times()
        cpu_time_used = sum(final_cpu_time) - sum(initial_cpu_time)
        elapsed_time = final_time - initial_time

        print("CPU Time Used: {:.2f} seconds".format(cpu_time_used))
        print("Elapsed Time: {:.2f} seconds".format(elapsed_time))
        print("CPU Usage: {:.2f}%".format(cpu_time_used / elapsed_time * 100))

    def test_internet_speed(self):
        import speedtest
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1024 / 1024  # convert to Mbps
        upload_speed = st.upload() / 1024 / 1024  # convert to Mbps
        ping = st.results.ping

        print("Download Speed: {:.2f} Mbps".format(download_speed))
        print("Upload Speed: {:.2f} Mbps".format(upload_speed))
        print("Ping: {} ms".format(ping))

    def test_memory(self):
        import psutil
        initial_memory = psutil.virtual_memory().used
        large_list = [0] * 10 ** 7
        final_memory = psutil.virtual_memory().used
        memory_used = final_memory - initial_memory
        print("Memory Used: {:.2f} MB".format(memory_used / 1024 / 1024))

    def get_cpu_usage(self, pid):
        import psutil
        for process in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            if process.pid == pid:
                return process.cpu_percent(interval=1)
        return None

    def get_average_cpu_usage_over_time(self, pid, record_time, finished_callback=None):
        import time
        usage_arr = []
        for _ in range(record_time):
            cpu_usage = self.get_cpu_usage(pid)
            usage_arr.append(cpu_usage)
            time.sleep(1)
        if finished_callback is not None:
            finished_callback(sum(usage_arr) / len(usage_arr))
        return sum(usage_arr) / len(usage_arr)

    class TestSingleSelenium:
        # Class variables
        #SELENIUM_INSTANCES_CPU_USAGE = []
        Lock = threading.Lock()

        def __init__(self, record_time):
            self.record_time = record_time
            #self.cpu_result = None

       #def test_finished_callback(self, cpu):
            #self.cpu_result = cpu
            #with self.__class__.Lock:
                #self.__class__.SELENIUM_INSTANCES_CPU_USAGE.append(cpu)

        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            try:
                #instance.record_time = args[0]  # Set record_time
                instance.cpu_result = None  # Initialize cpu_result to None
                driver = webdriver.Chrome(constants.ChromeConfig(True))
                #temp_bench = Benchmark()
                #cpu_thread = threading.Thread(target=temp_bench.get_average_cpu_usage_over_time,
                #                              daemon=True,
                #                              args=(driver.service.process.pid,
                #                                    int(args[0]),
                #                                    instance.test_finished_callback)
                #                             )
                #cpu_thread.start()
                for _ in range(args[0]):
                    for __ in range(4):
                        try:
                            driver.get(random_stress_test_websites())
                        except:
                            pass
                        time.sleep(0.25)
                #cpu_thread.join()
                #while instance.cpu_result is None:
                 #   time.sleep(0.1)
                driver.quit()
                #return instance.cpu_result  # Return cpu_result after it's set
            except Exception as e:
                print(f"An error occurred: {e}")
                return None

    def monitor_cpu(self, interval=10, duration=60, cpu_cutoff=None):
        import time
        import psutil
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=interval)
            print(f"CPU Usage: {cpu_percent}%")
            self.CPU_USAGE = cpu_percent
            if cpu_cutoff is not None and cpu_percent > cpu_cutoff:
                return cpu_percent

    def test_multiple_selenium(self, record_time=60, instances=3, step=2, find_max=False):
        import time
        kill_all_selenium_instances()
        test_threads = []
        count = instances
        for x in range(25):
            for _ in range(count):
                test_thread = threading.Thread(target=self.TestSingleSelenium, daemon=True, args=(record_time,))
                test_thread.start()
                test_threads.append(test_thread)
            find_max_thread = threading.Thread(target=self.monitor_cpu, daemon=True, args=(10, record_time, 80))
            find_max_thread.start()
            time.sleep(0.5)
            while find_max_thread.is_alive():
                instance_arr = get_selenium_instance_pids()
                if instance_arr:
                    print(f"Average: {sum(instance_arr) / len(instance_arr)}")
                    for pid in instance_arr:
                        print(f"PID {pid}:", self.get_cpu_usage(pid))
                time.sleep(1)
            for test_thread in test_threads:
                test_thread.join()
            max_threads = len(test_threads)
            test_threads = []
            kill_all_selenium_instances()
            if not find_max or self.CPU_USAGE > 80:
                average_instances_cpu_usage = (sum(self.SELENIUM_INSTANCES_CPU_USAGE) /
                                               len(self.SELENIUM_INSTANCES_CPU_USAGE))
                result = {"Max Instances": max_threads, "Average Instance CPU": average_instances_cpu_usage,
                        "CPU Usage": self.CPU_USAGE}
                self.SELENIUM_INSTANCES_CPU_USAGE.clear()
                self.CPU_USAGE = 0
                return result
            count += step


def is_valid_url(url):
    import re
    from urllib.parse import urlparse
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return bool(re.match(url_pattern, url)) and urlparse(url).scheme != ''


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def generate_random_string(length):
    import random
    import string
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def log(value):
    from constants import DEBUG_PRINT
    if DEBUG_PRINT:
        print(value)


def split_array_into_chunks(arr, chunk_size):
    return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]


def split_array_near_evenly(arr, num_chunks):
    #if not isinstance(list(arr), list):
     #   list(arr)
    if len(arr) == 0:
        return None
    length = len(arr)
    chunk_size = (length + num_chunks - 1) // num_chunks
    return [arr[i:i+chunk_size] for i in range(0, length, chunk_size)]


def full_scroll(driver):
    import time
    page_height = driver.execute_script("return document.body.scrollHeight;")
    driver.execute_script(f"window.scrollBy(0, {page_height});")
    time.sleep(0.5)
    driver.implicitly_wait(5)


def check_similarity(target_image_path, closest_image_path):
    import cv2
    from skimage.metrics import structural_similarity as compare_ssim
    target_image = cv2.imread(target_image_path)
    closest_image = cv2.imread(closest_image_path)

    target_image_resized = cv2.resize(target_image, (600, 600))  # Resize to the same size
    closest_image_resized = cv2.resize(closest_image, (600, 600))  # Resize to the same size

    target_image_resized = cv2.cvtColor(target_image_resized, cv2.COLOR_BGR2RGB)
    closest_image_resized = cv2.cvtColor(closest_image_resized, cv2.COLOR_BGR2RGB)

    score = compare_ssim(target_image_resized, closest_image_resized, multichannel=True, channel_axis=2)

    return score > 0.85


def enhance_image_quality():
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    dir = f"./{search_term}"
    upscaled_dir = f"./{search_term}_upscaled_2x"
    if not os.path.exists(upscaled_dir):
        os.makedirs(upscaled_dir)

    image_names = os.listdir(dir)

    for filename in image_names:
        start_time = time.time()
        image_path = os.path.join(dir,filename)
        upscaled_path = os.path.join(upscaled_dir, filename)
        image = Image.open(image_path)
        inputs = ImageLoader.load_image(image)
        preds = model(inputs)
        ImageLoader.save_image(preds, upscaled_path)
        print("Enhance Time: ", time.time() - start_time)
        exit()
        #ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')


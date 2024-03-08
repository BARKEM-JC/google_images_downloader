import atexit
import multiprocessing
import pickle
import threading

import easyocr
import cv2
import numpy as np
import selenium
from bs4 import BeautifulSoup
from pytesseract import pytesseract
import imagehash
import selenium.common
from PIL import Image
import os
import time
import requests
from selenium import webdriver
from selenium.common import TimeoutException, StaleElementReferenceException
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import lib
import constants

class ImageData():
    def __init__(self):
        self.id = lib.generate_random_string(15)
        self.hash = None
        self.path = None
        self.keyword = None
        self.upscaled = None
        self.URL = None
        self.src = None
        self.href = None


class SearchConfig():
    """
    :param exclude_keywords - A list of words that you want to exclude from results
    :param specific_sites - a list of websites to only use/include in the search
    :param safe_search - By default safe search is off
    :param loaded_limiter - The max amount of loaded elements at one time, stops elements from becoming stale
    :param grabdownload_threads - Amount of threads to grab the full image & download
    :param image_limit - the target amount or max limit of images to be downloaded in one session, is not always exact,
    I recommend keeping it low and instead just running the program in batched mode, the most i have gotten without
    too many issues was 300
    :param humans_only - It will only keep images were humans are detected
    :param humans_threshold - Use this to adjust how sensitive human detection is
    :param ignore_text - It will discard images that contain a specific amount of characters
    :param ignore_text_character_limit - anything above this limit will be discarded (if ignore_text is enabled)
    :param remove_duplicates - It will remove any duplicate images, which may include modified versions of the same
    image
    :param remove_duplicates_threshold - this will control how similar a image has to be to treat it as a duplicate
    :param remove_duplicates_iterations - images are first compared using hashes, this allows to check for multiple
    duplicates (May be removed in the future)
    :param remove_duplicates_only_ssim - This setting will not use hashes & instead do a Similarity Index on every
    image, very slow, can be up to 2 seconds per image
    :param sort_filenames - This will rename each file to use a number system instead of random characters
    :param super_resolution - Uses AI to upscale the images to a higher resolution, very slow, 2-4+ seconds per image.
    :param chrome_options - in case you want to use custom options
    :param related - gets top level related images aswell
    :param related_max - the maximum amount of related images on top of total_images, if set to 0 then it will count towards the total_images
    """
    def __init__(self,
                 exclude_keywords: list[str]=[],
                 specific_sites: list[str]=[],
                 safe_search: bool = False,
                 loaded_limiter: int = 50,
                 grabdownload_threads: int = 50,
                 scraper_processes: int = 5,
                 image_limit: int = 100,
                 humans_only: bool = False,
                 humans_threshold: float = 0.8,
                 ignore_text: bool = False,
                 ignore_text_character_limit: int = 10,
                 remove_duplicates: bool = True,
                 remove_duplicates_threshold: float = 0.85,
                 remove_duplicates_iterations: int = 2,
                 remove_duplicates_only_ssim: bool = False,
                 sort_filenames: bool = True,
                 super_resolution: bool = False,
                 chrome_options: webdriver.ChromeOptions = None,
                 related = False,
                 related_max = 0,
                 continue_after_processing_if_under_limit = False,
                ):
        self.exclude_keywords = exclude_keywords
        self.specific_sites = specific_sites
        self.safe_search = safe_search
        self.loaded_limiter = loaded_limiter
        self.grabdownload_threads = grabdownload_threads
        self.scraper_processes = scraper_processes
        self.image_limit = image_limit
        self.humans_only = humans_only,
        self.humans_threshold = humans_threshold,
        self.ignore_text = ignore_text,
        self.ignore_text_character_limit = ignore_text_character_limit,
        self.remove_duplicates = remove_duplicates,
        self.remove_duplicates_threshold = remove_duplicates_threshold,
        self.remove_duplicates_iterations = remove_duplicates_iterations,
        self.remove_duplicates_only_ssim = remove_duplicates_only_ssim,
        self.sort_filenames = sort_filenames,
        self.super_resolution = super_resolution,
        self.chrome_options = chrome_options,
        self.related = related,
        self.related_max = related_max
        self.continue_after_processing_if_under_limit = continue_after_processing_if_under_limit

    def get_query_params(self):
        return f"{self.specific_site_builder()}{self.exclude_keywords_builder()}"

    def specific_site_builder(self):
        query_string = ""
        for idx, site in enumerate(self.specific_sites):
            if not lib.is_valid_url(site):
                continue
            query_string += f"site:{site}"
            if idx < len(self.specific_sites) - 1:
                query_string += " OR "
        return query_string

    def exclude_keywords_builder(self):
        query_string = ""
        for idx, keyword in enumerate(self.exclude_keywords):
            query_string += f"-{keyword}"
            if idx < len(self.exclude_keywords) - 1:
                query_string += " AND "
        return query_string


class Search():
    def __init__(self,
                 SearchTerm: str,
                 Config = SearchConfig()
                 ):
        self.SearchTerm = SearchTerm
        self.Config = Config
        self._SessionItems = []

        self.MultiProcessingManager = multiprocessing.Manager()
        self.shared_dict = self.MultiProcessingManager.dict()
        self.shared_lastscrolly = multiprocessing.Value('i', 0)
        self.shared_lastthumbnailindex = multiprocessing.Value('i', 0)
        self._related_image_count = multiprocessing.Value('i', 0)
        self.MultiProcessingKillEvent = multiprocessing.Event()

        self.Dir = f'./GoogleImages/{self.SearchTerm}'
        if not os.path.exists(self.Dir):
            os.makedirs(self.Dir)
        self._DataFile = f'./GoogleImages/{self.SearchTerm}/{self.SearchTerm}_data.pkl'

    def __getstate__(self):
        state = self.__dict__.copy()
        state['shared_dict'] = dict(state['shared_dict'])
        state['shared_lastscrolly'] = int(state['shared_lastscrolly'].value)
        state['shared_lastthumbnailindex'] = int(state['shared_lastthumbnailindex'].value)
        state['_related_image_count'] = int(state['_related_image_count'].value)
        del state['MultiProcessingKillEvent']
        del state['MultiProcessingManager']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.shared_dict = self.MultiProcessingManager.dict(state['shared_dict'])
        self.shared_lastscrolly = multiprocessing.Value('i', state['shared_lastscrolly'])
        self.shared_lastthumbnailindex = multiprocessing.Value('i', ['shared_lastthumbnailindex'])
        self._related_image_count = multiprocessing.Value('i', ['_related_image_count'])
        self.MultiProcessingKillEvent = multiprocessing.Event()
        self.MultiProcessingManager = multiprocessing.Manager()

    def _save(self):
        try:
            with open(self._DataFile, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)

    def load(self):
        _data_file = os.path.join(f'./GoogleImages/{self.SearchTerm}', f"{self.SearchTerm}_data.pkl")
        if not os.path.exists(_data_file):
            return self
        with open(_data_file, 'rb') as f:
            return pickle.load(f)

    def StartThreads(self):
        # self.Config.continue_after_processing_if_under_limit
        print("Scraping Images")
        processes = []
        for i in range(self.Config.scraper_processes):
            p = multiprocessing.Process(target=WorkerProcess, daemon=True, args=(
                self.Config,
                self.SearchTerm,
                self.shared_dict,
                self.shared_lastscrolly,
                self.shared_lastthumbnailindex,
                self._related_image_count,
                self.MultiProcessingKillEvent,
            ))
            processes.append(p)
            p.start()
        last_length = 0

        while (len(self.shared_dict.keys()) -
               (self._related_image_count.value if self.Config.related_max != 0 else 0)
               < self.Config.image_limit):
            time.sleep(0.01)
            images_length = len(self.shared_dict.keys())
            if images_length > last_length:
                print(images_length)
                last_length = images_length
            if images_length % (self.Config.image_limit / 10) == 0:
                self._SessionItems = {key: self.shared_dict[key] for key in self.shared_dict.keys()}
                self._save()
            if not any(processor.is_alive() for processor in processes):
                break
        self._SessionItems = {key: self.shared_dict[key] for key in self.shared_dict.keys()}
        self.MultiProcessingKillEvent.set()
        for process in processes:
            process.join()
        processes.clear()

        self._save()

        threads = []
        max_threads = len(self._SessionItems.keys())

        print("Getting Full Image Sources & Downloading")
        thread_count = min(max_threads, self.Config.grabdownload_threads)
        items = list(self._SessionItems.values())
        items = lib.split_array_near_evenly(items, thread_count)
        thread_count = min(thread_count, len(items))
        for _ in range(thread_count):
            grabbing_thread = threading.Thread(target=self.GrabDownloadThread, daemon=True, args=(items[_],))
            threads.append(grabbing_thread)
            grabbing_thread.start()
        for thread in threads:
            thread.join()
        threads.clear()

        if items:
            self._SessionItems = {item.id: item for item_list in items for item in item_list}
            self._save()

        DataProcessor = DataPreProcessing(list(self._SessionItems.values()))
        DataProcessor.ValidateSessionImageDatas()
        if self.Config.remove_duplicates:
            DataProcessor.FindDuplicates()
        if self.Config.ignore_text:
            DataProcessor.detect_text()
        if self.Config.humans_only:
            DataProcessor.detect_humans()

    def GrabDownloadThread(self, workload):
        for item in workload:
            if not isinstance(item, ImageData):
                continue
            try:
                response = requests.get(item.href, stream=True)
                if response.status_code == 200:
                    html_content = response.text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    specific_element = soup.select_one("img[alt*='See original image']")
                    if specific_element:
                        item.URL = specific_element.get('src')
                else:
                    continue
            except:
                continue
            if item.URL:
                try:
                    response = requests.get(item.URL, stream=True)
                    if response.status_code == 200:
                        path = f'{self.Dir}/{self.SearchTerm.replace(" ", "_")}_{item.id}.jpg'
                        item.path = path
                        with open(path, 'wb') as f:
                            f.write(response.content)
                except:
                    pass


def CreateDriver(Config, SearchTerm):
    driver: webdriver = None
    try:
        driver = webdriver.Chrome(options=constants.ChromeConfig() if Config.chrome_options is None
                                  else Config.chrome_options[0])
        driver.get(
            f"https://www.google.com/search?tbm=isch&q="
            f"{SearchTerm}{Config.get_query_params()}"
            f"{'' if Config.safe_search else '&safe=off'}")
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, '//a[@data-nav="1"]')))
        return driver
    except TimeoutException:
        lib.log("Page load timeout")
    driver.quit()
    return None


def GetAttributes(element):
    attributes = {'id': None, 'role': None, 'href': None, 'src': None, 'alt': None, 'tag': None}
    try:
        attributes['id'] = element.get_attribute('role')
    except:
        pass
    try:
        attributes['role'] = element.get_attribute('role')
    except:
        pass
    try:
        attributes['href'] = element.get_attribute('href')
    except:
        pass
    try:
        attributes['src'] = element.get_attribute('src')
    except:
        pass
    try:
        attributes['alt'] = element.get_attribute('alt')
    except:
        pass
    try:
        attributes['tag'] = element.tag_name
    except:
        pass

    return attributes


def WorkerProcess(Config, SearchTerm, shared_dict, shared_lastscrolly, last_thumbnail_index, related_image_count, exit_flag):
    def ClickElement(element):
        driver.execute_script("arguments[0].scrollIntoView();", element)
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable(element))
            element.click()
            return True
        except:
            return False

    def GetData(thumbnail):
        try:
            attr = GetAttributes(thumbnail)
            if attr['role'] == 'button':
                thumbnail.click()
                attr = GetAttributes(thumbnail)
                print(attr)
            else:
                return False
            try:
                thumbnail_id = hash(attr['href'])
            except:
                return False
            if thumbnail_id in shared_dict:
                return False
            if attr['href'] is None:
                return False
            new_data = ImageData()
            new_data.id = thumbnail_id
            new_data.keyword = SearchTerm
            new_data.src = attr['src']
            new_data.href = attr['href']
            shared_dict[thumbnail_id] = new_data
            return True
        except selenium.common.StaleElementReferenceException:
            lib.log("Stale thumbnail element")
        except TimeoutException:
            lib.log("Thumbnail timeout")
        except Exception as e:
            print(e)
            pass
        return False

    # Create webdriver, retry for a few attempts, if exit flag is set then abort process
    driver = None
    try:
        for i in range(5):
            driver = CreateDriver(Config, SearchTerm)
            if driver:
                break
            if exit_flag.is_set():
                return
    except Exception as e:
        print(e)
        return
    try:
        driver.execute_script("window.scrollTo(0, arguments[0]);", shared_lastscrolly.value)
    except:
        return
    # Move webdriver page to correct position
    while not exit_flag.is_set():
        try:
            thumbnails = WebDriverWait(driver, 0.5).until(EC.visibility_of_all_elements_located((By.XPATH, f'(//a[@data-nav="1"])[position() > {last_thumbnail_index.value}]')))
            with last_thumbnail_index.get_lock():
                last_thumbnail_index = last_thumbnail_index.value + len(thumbnails)
        except TimeoutException:
            lib.full_scroll(driver)
            print("Timeout waiting for images")
            continue
        except:
            continue
        for thumbnail in thumbnails:
            if exit_flag.is_set():
                break
            if not GetData(thumbnail):
                continue

        button = driver.find_element(By.CSS_SELECTOR, 'input[type="button"][value="Show more results"]')
        if button.is_displayed():
            try:
                button = WebDriverWait(driver, 2).until(EC.element_to_be_clickable(button))
                button.click()
                time.sleep(1.5)
                lib.full_scroll(driver)
            except TimeoutException:
                pass
        else:
            try:
                element = driver.find_element(By.CSS_SELECTOR, "div:contains('Looks like you\'ve reached the end')")
                if element.is_displayed():
                    break
            except:
                pass
        lib.full_scroll(driver)
        try:
            with shared_lastscrolly.get_lock():
                shared_lastscrolly = driver.execute_script("return window.pageYOffset")
        except:
            pass
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    driver.quit()


def highlight_element(driver, element):
    driver.execute_script("arguments[0].style.border='20px solid red'", element)

class Hasher():
    def __init__(self):
        pass

    def image_hash(self, image_path):
        try:
            img = Image.open(image_path)
            hash_value = imagehash.average_hash(img)
            return hash_value
        except:
            return None

    def _hamming_distance(self, hash1, hash2):
        return hash1 - hash2

    def find_closest_hash(self, target_hash, hash_list):
        closest_hash = None
        min_distance = float('inf')
        for i, hash_value in enumerate(hash_list):
            distance = self._hamming_distance(target_hash, hash_value)
            if distance < min_distance:
                min_distance = distance
                closest_hash = i
        return closest_hash

    def find_closest_hashes(self, source_hash, hash_list, num_closest=3):
        distances = [(target_hash, self._hamming_distance(source_hash, target_hash)) for target_hash in
                     hash_list]
        closest = sorted(distances, key=lambda x: x[1])[:num_closest]
        for close in reversed(closest):
            close_hash, close_score = close
            if close_score > 15 or source_hash == close_hash:
                closest.remove(close)
        return closest


class DataPreProcessing():
    def __init__(self,
                 SessionImageDatas: list[ImageData],
                 ):
        self._SessionImageDatas = SessionImageDatas
        self._Hasher = Hasher()
        self._HashImages()

    def __RemoveImageData(self, img, keep_instead=False):
        if keep_instead:
            print("Duplicate/Human/Text: ", img.path)
            return
        try:
            os.remove(img.path)
        except Exception as e:
            print(e)
            pass
        try:
            self._SessionImageDatas.remove(img)
        except:
            pass

    def _HashImages(self):
        print("Hashing Images")
        for img in self._SessionImageDatas:
            if img.hash is None and img.path is not None:
                img.hash = self._Hasher.image_hash(img.path)

    def FindDuplicates(self):
        print("Finding Duplicates")
        HashMap = {}
        for img in reversed(self._SessionImageDatas):
            if img.hash not in HashMap:
                HashMap[img.hash] = img
            else:
                self.__RemoveImageData(img)

        HashList: list = list(HashMap.keys())
        for i in range(len(HashList) - 1):
            if i > len(HashList) - 1:
                break
            closest_hashes = self._Hasher.find_closest_hashes(HashList[i], HashList, 5)
            for close_hash, close_score in closest_hashes:
                if lib.check_similarity(HashMap[HashList[i]].path, HashMap[close_hash].path):
                    HashList.remove(close_hash)
                    self.__RemoveImageData(HashMap[close_hash])

    def ValidateSessionImageDatas(self, keep_instead_of_delete=False):
        print("Validating Session Image Datas")
        for img in reversed(self._SessionImageDatas):
            if img is None or img.URL is None or img.path is None or not os.path.exists(img.path):
                self.__RemoveImageData(img, keep_instead_of_delete)
                continue
            try:
                image = cv2.imread(img.path)
                if image is None or not Image.open(img.path):
                    self.__RemoveImageData(img, keep_instead_of_delete)
                    continue

                height, width, channels = image.shape
                if height == 0 or width == 0:
                    self.__RemoveImageData(img, keep_instead_of_delete)
                    continue
            except Exception as e:
                self.__RemoveImageData(img, keep_instead_of_delete)

    def detect_humans(self):
        print("Detecting Humans")
        confidence_threshold = 0.5
        nms_threshold = 0.3
        filtered_images = []
        classes = open('coco.names').read().strip().split('\n')
        net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        ln = net.getLayerNames()
        # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #FOR CUDA / GPU
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        for ImageData in self._SessionImageDatas:
            img = cv2.imread(ImageData.path)
            one_human = False
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            r = blob[0, 0, :, :]

            net.setInput(blob)
            t0 = time.time()
            outputs = net.forward(ln)
            t = time.time()

            boxes = []
            confidences = []
            classIDs = []
            h, w = img.shape[:2]

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.5:
                        box = detection[:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        box = [x, y, int(width), int(height)]
                        boxes.append(box)
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        if classes[classID] == 'person':
                            one_human = True

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in colors[classIDs[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.imshow("Detect humans: ", img)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

            if not one_human:
                self.__RemoveImageData(ImageData)

    def detect_text(self):
        print("Detecting Text")
        reader = None
        result = None
        if constants.OCR_METHOD == constants.OCR.EASYOCR:
            reader = easyocr.Reader(['ch_sim', 'en'])
        character_threshold = 10
        for img in reversed(self._SessionImageDatas):
            if constants.OCR_METHOD == constants.OCR.EASYOCR:
                try:
                    result = reader.readtext(img.path, detail=0)
                except Exception as e:
                    print(e)
                    continue
            if constants.OCR_METHOD == constants.OCR.TESSERACT:
                for _ in range(2):
                    try:
                        result = (pytesseract.image_to_string(Image.open(img.path), config=r'--psm 11', timeout=5)
                               .strip().strip("\n").strip(" "))
                    except Exception as e:
                        print(e)
            if result is not None and len(result) > character_threshold:
                self.__RemoveImageData(img)
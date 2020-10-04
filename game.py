from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import time

class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self.browser = webdriver.Chrome(executable_path = r".\chrome-driver\chromedriver.exe",chrome_options=chrome_options)
        # self.browser.set_window_position(x=0,y=0)
        self.browser.get('chrome://dino')
        self.browser.implicitly_wait(2)
        self.browser.maximize_window()
        time.sleep(1)

    def get_crashed(self):
        return self.browser.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self.browser.execute_script("return Runner.instance_.playing")
    def restart(self):
        self.browser.execute_script("Runner.instance_.restart()")
        
    def press_up(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN) 
    def press_right(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_RIGHT)

    def start_game(self):
        self.press_up()

    def get_score(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def get_highscore(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.highScore")
        for i in range(len(score_array)):
            if score_array[i] == '':
                break
        score_array = score_array[i:]        
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self.browser.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self.browser.execute_script("return Runner.instance_.play()")
    def end(self):
        self.browser.close()

    def play_game(self):
        self.start_game()



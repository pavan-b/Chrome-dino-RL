from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import time

chrome_options = Options()
chrome_options.add_argument("disable-infobars")
chrome_options.add_argument("--mute-audio")
browser = webdriver.Chrome(executable_path = r".\chrome-driver\chromedriver.exe", chrome_options=chrome_options)
browser.implicitly_wait(2)
browser.set_window_position(x=-10,y=0)
browser.maximize_window()
browser.get('chrome://dino')
# wait for the page to load
time.sleep(1)

elem=browser.find_element_by_tag_name("body")
elem.send_keys(Keys.ARROW_UP)

t=browser.execute_script("return Runner.instance_.crashed")
# c=0
while not t:
    # c+=1
    # elem.screenshot(str(c)+".png")
    t=browser.execute_script("return Runner.instance_.crashed")
    elem.send_keys(Keys.ARROW_UP)

    print("score : ",browser.execute_script("return Runner.instance_.distanceMeter.digits"))
print("highscore : ",browser.execute_script("return Runner.instance_.distanceMeter.highScore"))


browser.close()

    

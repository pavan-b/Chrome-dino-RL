from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np

import time

from game import Game

class Capture:
    def __init__(self):
        pass

    def record_screen(self):

        game= Game()
        game.play_game()
        
        time.sleep(2)

        crashed = game.get_crashed()
        last_time=time.time()
        while( not  crashed):
            # control frame rate
            # time.sleep(.1)
            # The bounding box is a (left_x, top_y, right_x, bottom_y) tuple
            printscreen =  np.array( ImageGrab.grab(bbox=(0,255,1365,515)))
            print("time : {} seconds".format(time.time()-last_time))
            # cv2.imshow('window',printscreen)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break

            # to save as image
            # im = Image.fromarray(printscreen)
            # im.save(str(time.time())+".jpeg")

            last_time=time.time()
            crashed=game.get_crashed()
            return printscreen


        
def main():
    cp=Capture()
    return cp.record_screen()

if __name__ == "__main__":
    main()
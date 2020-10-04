import numpy as np
import time
from  capture  import Capture
import cv2
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from PIL import ImageGrab

from game import Game


class Dino():
    def __init__(self,frame_rate=0,skip_frame=0):
        self.frame_rate=frame_rate
        self.skip_frame=skip_frame

        self.action_space=2
        self.observation_space=(65,33)

    def start(self):
        self.game= Game()
        self.game.play_game()
        
        time.sleep(2)

    def preprocess(self,img):
        # ##original/4
        img=cv2.resize(img,self.observation_space,interpolation=cv2.INTER_AREA)
        return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),axis=1)

        # #####to save as image
        # img=cv2.resize(img,self.observation_space,interpolation=cv2.INTER_AREA)
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = Image.fromarray(img.astype(np.uint8))
        # img.save(str(time.time())+".jpeg")

    def record_screen(self):   
        # 260x130
        printscreen =  np.array( ImageGrab.grab(bbox=(50,360,310,490)))        
        return printscreen

    def crashed(self):
        return self.game.get_crashed()
        
    def restart(self):
        return self.game.restart()

    def get_score(self):
        return self.game.get_score()
    
    def get_highscore(self):
        return self.game.get_highscore()

    def step(self,action):
        """
        skip frame
        action:
            0:do nothing
            1:jump
            2:crouch
        return:
            next_state, reward, done
        """
        penality=2
        if(action == 0):
            penality=0
        elif(action == 1):
            self.game.press_up()
        else:
            self.game.press_down()  

        new_state=self.get_state()
        #skip_frame
        frame=-1
        while(frame <self.skip_frame):
            frame+=1
            new_state=self.get_state()

        done =self.crashed()
        high_score=self.get_highscore()
        score=self.get_score()
        # make the agent walk before the obsticles appear
        reward=penality
        if not done:    
            if high_score>0:
                if score>high_score:
                    reward=(0.1*score)/1+penality
                else:
                    reward=(0.1*score)/5+penality
        else:
            reward=(0.1*score)/-10+penality   
        if score>200:
            reward=100
            done=True

        return new_state, reward, done
    
    def get_state(self):
        img=self.record_screen()
        return self.preprocess(img)

    def reset(self):
        self.restart()
        return self.get_state()

    def run(self):
        capture=0
        frame=-1
        start=time.time()
        crashed=self.crashed()
        while(not crashed):
            if frame >=self.skip_frame:
                frame=-1
                img=self.record_screen()
                self.preprocess(img)
                capture+=1
                continue
            else:
                img=self.record_screen()
                self.preprocess(img)
                frame+=1
               
            crashed=self.crashed()
                ####self.restart()
                ####self.run()
        stop=time.time()-start
        print(stop," secs")
        print(capture," total captures")
        print(capture/stop," frames per sec")

def main():
    d=Dino(skip_frame=0)
    d.start()
    d.run()

if __name__ == "__main__":
    main()

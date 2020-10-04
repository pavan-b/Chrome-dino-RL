import numpy as np
from collections import deque
from dqn import DQN
from memory import Memory
from env import Dino
import tensorflow as tf
import time 
import os
import shutil
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage

class Agent():
    def __init__(self,stack_size=3,episodes=500,max_steps=20,mem_size=100000,batch_size=64):
        self.env=Dino(skip_frame=0)
        self.action_space=self.env.action_space
        self.observation_space=self.env.observation_space
        self.episodes=episodes
        self.max_steps=max_steps
        self.mem_size=mem_size
        self.batch_size=batch_size

         # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001      
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.999
        # self.epsilon_min = 0.01
        self.train_start = 500


      
        self.skip_frame=self.env.skip_frame
        self.stack_size = stack_size # We stack 4 frames

        # Initialize deque with zero-images one array for each image
        self.stacked_frames  =  deque([np.zeros(self.observation_space, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    def stack_frames(self, state, is_new_episode):
        """
        """
        frame=np.reshape(state,self.observation_space)
        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames  =   deque([np.zeros(self.observation_space, dtype=np.int) for i in range(self.stack_size)], maxlen=self.stack_size)
            # Because we're in a new episode, copy the same frame 4x
            for _ in range(self.stack_size):
                self.stacked_frames.append(frame)
                
            
            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)
            # print(stacked_state.shape," <---")
            
        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2) 
        
        return stacked_state
    
    def run(self):
        if os.path.exists("tensorboard"):
            shutil.rmtree(os.path.abspath("tensorboard"), ignore_errors=True)

        # get size of state and action from environment
        state_size = self.observation_space
        action_size = self.action_space

        memory=Memory(self.mem_size)

        agent = DQN(state_size, action_size,memory)
        model=agent.build_model(self.batch_size,self.stack_size)
        # writer=agent.setup_tensorboard()

        scores, episodes, crashes, captures = [], [], [], []
        env=self.env
        env.start()
        time.sleep(2)
        crash=0

        
        for e in range(1,self.episodes):
            done = False
            score = 0
            step=0
            capture=0
            crash=0
            training_time=0
            start_time=time.time()
            state = env.reset()
            
            # Remember that stack frame function also call our preprocess function.
            state = self.stack_frames( state,True)

            if e>1:
                crashes.append(crash)
                
            while crash < self.max_steps:
                if step>0:
                    env.restart()
                step+=1
                
            
                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                capture+=1

                if done:
                    crash+=1
                   # The episode ends so no next state
                    next_state = np.zeros(self.observation_space, dtype=np.int)
                    
                    next_state = self.stack_frames( next_state, False)

                    # Set step = max_steps to end the episode
                    step = self.max_steps
                
                    scores.append(score)
                    # pylab.plot(episodes, scores, 'b')
                    # pylab.savefig("./save_graph/cartpole_dqn.png")
                    # print(agent.memory)
                    

                    # Add experience to memory
                    agent.append_sample((state, action, reward, next_state, done))
                    
                    score=0
                    # ##########################Learning start############################
                    if len(agent.memory.memory) > self.batch_size:
                        train_time_start=time.time()
                        ### LEARNING PART            
                        # Obtain random mini-batch from memory
                        batch = agent.sample_from_mem(self.batch_size)
                        states_mb = np.array([each[0] for each in batch]).reshape(self.batch_size,*self.observation_space,self.stack_size)
                        actions_mb = np.array([each[1] for each in batch])
                        rewards_mb = np.array([each[2] for each in batch]) 
                        next_states_mb = np.array([each[3] for each in batch]).reshape(self.batch_size,*self.observation_space,self.stack_size)
                        dones_mb = np.array([each[4] for each in batch])
                        
                        # target_Qs_batch =np.zeros((self.batch_size,self.action_space))
                        target_Qs_batch =model.predict(states_mb)

                        Qs_next_state=model.predict(next_states_mb)

                        # update the target values
                        for i in range(self.batch_size):
                            if dones_mb[i]:
                                target_Qs_batch[i][actions_mb[i]]=rewards_mb[i]
                            else: # non-terminal state
                                target = rewards_mb[i] + self.discount_factor * np.max(Qs_next_state[i])
                                # ###SARSA
                                # target = rewards_mb[i] + self.discount_factor * (Qs_next_state[i][actions_mb[i]])
                                target_Qs_batch[i][actions_mb[i]]=target
                        

                        log_dir = os.path.join("tensorboard","log") 
                        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
                        file_writer.set_as_default()
                        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                        # model fit
                        # model.fit(states_mb, targets_mb, epochs=1, verbose=1,callbacks=[tensorboard_callback],use_multiprocessing=True)
                        model.fit(states_mb, target_Qs_batch,batch_size=self.batch_size, epochs=1,callbacks=[tensorboard_callback],verbose=0,use_multiprocessing=True)
                        
                        training_time+=time.time()-train_time_start
                    # ##########################Learning end##############################

                else:
                    
                     # Stack the frame of the next_state
                    next_state = self.stack_frames( next_state, False)
                
                    # Add experience to memory
                    agent.append_sample((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state
                   
                    score += reward

            episodes.append(e)
            episodic_scores=[sum(scores[e-1:e+9]) for e in episodes ]
            stop_time=time.time()-(training_time)
            captures.append(capture)
            
            crashes.append(crash)

            score_per_epd=episodic_scores[e-1]/self.max_steps
            tf.summary.scalar('score', data=score_per_epd, step=e)
            tf.summary.scalar('epsilon', data=agent.epsilon, step=e)
            print("((((((((((((((((((((((((((((((=======================)))))))))))))))))))))))))))))")
            print("episode:", e, "  score:", score_per_epd, "  memory length:",
                        len(agent.memory.memory)," frs:",captures[e-1]/(stop_time-start_time), "  epsilon:", agent.epsilon)
            print("((((((((((((((((((((((((((((((=======================)))))))))))))))))))))))))))))")

            if e %5==0:
                agent.save_model(model)
                    
           

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    Agent().run()


               
    

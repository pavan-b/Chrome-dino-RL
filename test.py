import os
import shutil
import time
import numpy as np
import tensorflow as tf
from actorCritic import ActorCritic

def run():
    if os.path.exists("tensorboard"):
        shutil.rmtree(os.path.abspath("tensorboard"), ignore_errors=True)

    agent = ActorCritic()
    # get size of state and action from environment
    state_size = agent.observation_space
    action_size = agent.action_space



    scores, episodes, crashes, captures = [], [], [], []
    env=agent.env
    env.start()
    time.sleep(2)
    crash=0


    for e in range(1,agent.episodes):
        done = False
        score = 0
        step=0
        capture=0
        crash=0
        training_time=0
        start_time=time.time()
        state = env.reset()
        
        # Remember that stack frame function also call our preprocess function.
        state = agent.stack_frames( state,True)

        if e>1:
            crashes.append(crash)
            
        while crash < agent.max_steps:
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
                next_state = np.zeros(agent.observation_space, dtype=np.int)
                
                next_state = agent.stack_frames( next_state, False)

                # Set step = max_steps to end the episode
                step = agent.max_steps
            
                scores.append(score)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/cartpole_dqn.png")
                # print(agent.memory)
                

                # Add experience to memory
                agent.append_sample((state, action, reward, next_state, done))
                
                score=0

                train_time=agent.train_model()
                training_time+=train_time
              
            else:
                
                    # Stack the frame of the next_state
                next_state = agent.stack_frames( next_state, False)
            
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

        score_per_epd=episodic_scores[e-1]/agent.max_steps
        tf.summary.scalar('score', data=score_per_epd, step=e)
        print("((((((((((((((((((((((((((((((=======================)))))))))))))))))))))))))))))")
        print("episode:", e, "  score:", score_per_epd, "  memory length:",
                    len(agent.memory)," frs:",captures[e-1]/(stop_time-start_time))
        print("((((((((((((((((((((((((((((((=======================)))))))))))))))))))))))))))))")

        if e %5==0:
            agent.save_model()
                
        

if __name__ == "__main__":

    run()


               
    


import numpy as np
import tensorflow as tf

class DQN():
    def __init__(self,observation_size,action_size,memory):
        super().__init__()
        self.observation_size=observation_size
        self.action_size=action_size

        self.memory=memory

         # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.01      
        self.epsilon = 1.0
        self.epsilon_decay = 0.99991
        self.epsilon_min = 0.1
        self.train_start = 500
        
    def append_sample(self,exp):
        self.memory.add(exp)
    
    def sample_from_mem(self,batch_size):
        return self.memory.sample(batch_size)



    def build_model(self,batch_size,channels):
        '''
        TODO:
        Build multilayer perceptron to train the Q(s,a) function. In this neural network, the input will be states and the output 
        will be Q(s,a) for each (state,action). 
        Note: Since the ouput Q(s,a) is not restricted from 0 to 1, we use 'linear activation' as output layer.

        Loss Function:
        Loss=1/2 * (R_t + γ∗max Q_t (S_{t+1},a)−Q_t(S_t,a)^2
                which is 'mean squared error'

        '''     
        model = tf.keras.models.Sequential()          
        # Add a Convolutional layer  activation=tf.keras.layers.LeakyReLU(alpha=0.3)
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=[*self.observation_size,channels]))
        # Add a Max pooling layer
        model.add(tf.keras.layers.MaxPool2D())
        # Add a Convolutional layer
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        # Add a Max pooling layer
        model.add(tf.keras.layers.MaxPool2D())
        # Add a Convolutional layer
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
        # Add a Max pooling layer
        model.add(tf.keras.layers.MaxPool2D())
        # Add the flattened layer
        model.add(tf.keras.layers.Flatten())
        # Add the hidden layer
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        # Adding a dropout layer
        model.add(tf.keras.layers.Dropout(0.3))
        # Add the output layer
        model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))
        # Compiling the model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy",tf.keras.metrics.AUC()])
        print (model.summary())
        self.model= model
        return model
    

    # def setup_tensorboard():
    #     ##### To launch tensorboard : tensorboard --logdir=/tensorboard/loss

    #     # Setup TensorBoard Writer
    #     writer = tf.summary.FileWriter("/tensorboard/loss")

    #     ## Losses
    #     tf.summary.scalar("Loss", self.model.loss)

    #     write_op = tf.summary.merge_all()
    #     return writer
    
    def get_action(self, state):
        '''
        Select action
        Args:
            state: At any given state, choose action
        
        TODO:
        Choose action according to ε-greedy policy. We generate a random number over [0, 1) from uniform distribution.
        If the generated number is less than ε, we will explore, otherwise we will exploit the policy by choosing the
        action which has maximum Q-value.
        
        More the ε value, more will be exploration and less exploitation.
        
        '''
        # choose random action if generated random number is less than ε.
        # Action is represented by index, 0-Number of actions, like (0,1,2,3) for 4 actions
        
        if np.random.rand() <= self.epsilon:
            action= np.random.choice(self.action_size)
        # if generated random number is greater than ε, choose the action which has max Q-value
        else:
            q_value = self.model.predict(state.reshape((1,*state.shape)))
            action= np.argmax(q_value[0])

        if self.epsilon >self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return action
    
    

        
               
    

    def save_model(self,model):
        tf.keras.models.save_model(model,'./dqn_model.h5')

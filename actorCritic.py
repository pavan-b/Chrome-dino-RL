from env import Dino
from memory import Memory

import numpy as np
from collections import deque
import tensorflow as tf
import os
import shutil
import time
import random


class ActorCritic():
	def __init__(self,stack_size=4,episodes=500,max_steps=20,mem_size=100000,batch_size=64):
		self.env=Dino(skip_frame=0)
		self.action_space=self.env.action_space
		self.value_size=1
		self.observation_space=self.env.observation_space
		self.episodes=episodes
		self.max_steps=max_steps
		self.mem_size=mem_size
		self.batch_size=batch_size
		self.memory=deque(maxlen = mem_size)

		# These are hyper parameters for the DQN
		self.discount_factor = 0.99

		self.critic_lr=0.005
		self.actor_lr=0.001
		
		self.skip_frame=self.env.skip_frame
		self.stack_size = stack_size # We stack 4 frames

		# Initialize deque with zero-images one array for each image
		self.stacked_frames  =  deque([np.zeros(self.observation_space, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

		self.actor,self.critic=self.build_models(self.stack_size)

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

	def append_sample(self,exp):
		self.memory.append(exp)  


	def build_models(self,channels):
		actor = tf.keras.models.Sequential()          
		# Add a Convolutional layer  activation=tf.keras.layers.LeakyReLU(alpha=0.3)
		actor.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu",kernel_initializer='he_uniform', input_shape=(*self.observation_space,channels)))
		# Add a Max pooling layer
		actor.add(tf.keras.layers.MaxPool2D())
		# Add a Convolutional layer
		actor.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu",kernel_initializer='he_uniform'))
		# Add a Max pooling layer
		actor.add(tf.keras.layers.MaxPool2D())
		# Add a Convolutional layer
		actor.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu",kernel_initializer='he_uniform'))
		# Add a Max pooling layer
		actor.add(tf.keras.layers.MaxPool2D())
		# Add the flattened layer
		actor.add(tf.keras.layers.Flatten())
		# Add the hidden layer
		actor.add(tf.keras.layers.Dense(512, activation="relu",kernel_initializer='he_uniform'))
		# Adding a dropout layer
		actor.add(tf.keras.layers.Dropout(0.3))
		# Add the output layer
		actor.add(tf.keras.layers.Dense(self.action_space,kernel_initializer='he_uniform', activation='softmax'))
		# Compiling the model
		actor.compile(optimizer=tf.keras.optimizers.Adam(lr=self.actor_lr), loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
		print (actor.summary())

		critic = tf.keras.models.Sequential()
		# Add a Convolutional layer  activation=tf.keras.layers.LeakyReLU(alpha=0.3)
		critic.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu",kernel_initializer='he_uniform', input_shape=(*self.observation_space,channels)))
		# Add a Max pooling layer
		critic.add(tf.keras.layers.MaxPool2D())
		# Add a Convolutional layer
		critic.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu",kernel_initializer='he_uniform'))
		# Add a Max pooling layer
		critic.add(tf.keras.layers.MaxPool2D())
		# Add a Convolutional layer
		critic.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu",kernel_initializer='he_uniform'))
		# Add a Max pooling layer
		critic.add(tf.keras.layers.MaxPool2D())
		# Add the flattened layer
		critic.add(tf.keras.layers.Flatten())
		# Add the hidden layer
		critic.add(tf.keras.layers.Dense(256,activation='relu',kernel_initializer='he_uniform'))
		critic.add(tf.keras.layers.Dropout(0.3))
		critic.add(tf.keras.layers.Dense(512,activation='relu',kernel_initializer='he_uniform'))
		critic.add(tf.keras.layers.Dropout(0.3))
		critic.add(tf.keras.layers.Dense(self.value_size,activation='linear',kernel_initializer='he_uniform'))
		critic.compile(optimizer=tf.keras.optimizers.Adam(lr=self.critic_lr),loss='mse',metrics=['accuracy'])
		print(critic.summary())

		return actor,critic
	# using the output of policy network, pick action stochastically
	def get_action(self, state):
		policy = self.actor.predict(np.expand_dims(state,axis=0), batch_size=1).flatten()
		return np.random.choice(self.action_space, 1, p=policy)[0]

	# update policy network every episode
	def train_model(self):

	# ##########################Learning start############################
		if len(self.memory) > self.batch_size:
			train_time_start=time.time()
			### LEARNING PART            
			# Obtain random mini-batch from memory
			batch = random.sample(self.memory,self.batch_size)
			states_mb = np.array([each[0] for each in batch]).reshape(self.batch_size,*self.observation_space,self.stack_size)
			actions_mb = np.array([each[1] for each in batch])
			rewards_mb = np.array([each[2] for each in batch]) 
			next_states_mb = np.array([each[3] for each in batch]).reshape(self.batch_size,*self.observation_space,self.stack_size)
			dones_mb = np.array([each[4] for each in batch])

			target = np.full((self.batch_size, self.value_size),0)
			advantages = np.full((self.batch_size, self.action_space),0)

			value = self.critic.predict(states_mb)
			# print(value[0].shape,"<<---")
			next_value = self.critic.predict(next_states_mb)

			# update the target values
			for i in range(self.batch_size):
				if dones_mb[i]:
					advantages[i][actions_mb[i]] = rewards_mb[i] - value[i]
					target[i][0] = rewards_mb[i]
				else: # non-terminal state
					advantages[i][actions_mb[i]] = rewards_mb[i] + self.discount_factor * (next_value[i] - value[i])
					target[i][0] = rewards_mb[i] + self.discount_factor * next_value[i]
			

			log_dir = os.path.join("tensorboard","a2c") 
			file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
			file_writer.set_as_default()
			tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
			# model fit
			self.actor.fit(states_mb, advantages,batch_size=self.batch_size,callbacks=[tensorboard_callback], epochs=1, verbose=0,use_multiprocessing=True)
			self.critic.fit(states_mb, target,batch_size=self.batch_size,callbacks=[tensorboard_callback], epochs=1, verbose=0,use_multiprocessing=True)
			
			return time.time()-train_time_start
		else:
			return 0
		# ##########################Learning end##############################

	def save_model(self):
		tf.keras.models.save_model(self.actor,'./actor.h5')
		tf.keras.models.save_model(self.critic,'./critic.h5')


	

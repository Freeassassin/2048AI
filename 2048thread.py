import os
import sys
import gym
import time
import gym_2048
from time import sleep
import numpy as np
from numpy import array
from collections import Counter
from statistics import median, mean

InitGames = 10000
training_data = []
scoreReq = 3000
scores = []
accepted_scores = []

env = gym.make("2048-v0")
env.reset()

for i in range(InitGames):

    score = 0
    game_memory = []
    prev_observation = []

    done = False 
    while not done:
        #env.render()
        #sleep(0.1)
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        

        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
            prev_observation = []
        
       
        for j in np.nditer(observation, flags=['external_loop'], order='F'):
        
            for i in np.nditer(j):

                prev_observation.append(int(i))

        score+=reward
        if done: break


    if score >= scoreReq:
        accepted_scores.append(score)
        output = []
        for data in game_memory:
            if data[1] == 1:
                output = [0,1,0,0]
            elif data[1] == 0:
                output = [1,0,0,0]
            elif data[1] == 2:
                output = [0,0,1,0]
            elif data[1] == 3:
                output = [0,0,0,1]

            training_data.append([data[0], output])
    env.reset()
    scores.append(score)  

print('Average accepted score:',mean(accepted_scores))
print('Median score for accepted scores:',median(accepted_scores))
print(Counter(accepted_scores))

File_object = open(r"2048_train.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 
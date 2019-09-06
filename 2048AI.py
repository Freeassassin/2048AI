import os
import sys
import gym
import time
import gym_2048
import numpy as np
from time import sleep
import tensorflow as tf
from keras import layers
from collections import Counter
from statistics import median, mean
from keras.layers import Dense, Input
from keras.models import Sequential , load_model


training_data= []
training_dataFile = open("2048_train.txt", "r")
training_dataFile = training_dataFile.read()
exec(training_dataFile)

x_train = np.array([])
y_train = np.array([])
x = 0

for i in training_data:
    x += 1
    x_train = np.append(x_train, i[0])
    y_train = np.append(y_train, i[1])

x_train = x_train.reshape(x,16)
y_train = y_train.reshape(x,4)



model = Sequential([

    Dense(10, activation = 'relu', input_dim = 16),
    Dense(10, activation = 'relu'),
    Dense(4, activation = 'softmax'),
])

#model = load_model('2048(1).h5')
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train,y_train,epochs = 3)

model.save("2048(1).h5")
#model = load_model('my_model.h5')


InitGames = 1000
training_data = []
scoreReq = 0
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
        #os.system("cls")
        #env.render()
        #sleep(1)

        if not len(prev_observation) > 0 :

            action = env.action_space.sample()
        else:
            x_test = np.array([])

            for i in prev_observation:
                x_test = np.append(x_test, i)

            x_test = x_test.reshape(1,16)

            prediction = model.predict([x_test])
            
            action =np.argmax(prediction[0])        
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

"""
File_object = open(r"2048_train.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 
"""
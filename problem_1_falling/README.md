# The falling objects challenge

Your goal is to try to avoid whatever is coming at you. Can you think of the optimal agent 
policy? 

Any overlap of the agent (blue box) with a falling object results in a reward of -1. There is only 
one obstacle on the map at any given moment and they fall with constant speed. The 
obstacles can have different shapes, sizes, rotations, speeds. Your agent has to be prepared for 
any kind of obstacle. 

You have to implement an agent, just like our demo agent from `demo_agent.py`. The agent will be 
tested using the script `test_agent.py` using different config files. We will evaluate the agent using new configs and obstacle shapes, so try to have a agent that is prepared. There must be a better solution than the random agent.

Good luck!

## Getting Started

You can play the game by running the `play_game.py` script. Use the keys `["W", "A", "S", "D"]`
keys to control the agent.

## Submission Format
Your agent must be implemented in python with a class similar to `DemoAgent`. The policy of the 
agent will be implemented in the method `act` and will only have access to the variables returned by the `env.step` method: `observation, reward, done_state` (a numpy array containing the game screen image/ a float number representing the reward / a boolean representing end of episode if True). 

**A python file with your agent class that can be run using our example (details below).**

## Running the tests

```
python test_agent.py -a <module_name>+<class_name>
```
Example:
```
python test_agent.py -a demo_agent+DemoAgent
```

## Solution

The proposed solution is an agent trained using Deep Q Learning (based on this paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf and this tutorial: https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)

The trained model should be downloaded from: https://drive.google.com/open?id=1ujxv0vSll30n0_QQlH-8rmLTOyWOLFHj and saved to models/

## Results

### The DQN agent has been tested with different environment configurations (the names are quite self-descriptive). The average score for multiple runs are:

Config: bigger_agent, mean score in 10 simulations of 10000 steps: -920.00

Config: default, mean score in 10 simulations of 10000 steps: -304.10

Config: new_obstacle, mean score in 10 simulations of 10000 steps: -255.50

Config: speedy, mean score in 10 simulations of 10000 steps: -327.40

### The results of the random agent with the same configurations are:

Config: bigger_agent, mean score in 10 simulations of 10000 steps: -1701.40

Config: default, mean score in 10 simulations of 10000 steps: -914.80

Config: new_obstacle, mean score in 10 simulations of 10000 steps: -987.10

Config: speedy, mean score in 10 simulations of 10000 steps: -906.20

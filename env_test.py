
import metaworld
import random

print(metaworld.ML1_V3.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1_V3('reach-v3') # Construct the benchmark, sampling tasks

env = ml1.train_classes['reach-v3']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, _,_, info = env.step(a)
print(obs) 
print(reward)
print(info)
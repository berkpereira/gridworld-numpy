# Functions

## ```generate_episode(MDP, policy)```

We have previously implemented the relevant MDP in a sampled fashion. However, we will have to adapt policy functions to work that way too, probably using a similar approach (create array of probabilities corresponding to actions, fill the array with probabilities as given by the policy, then sample from that probability distribution and return the action).
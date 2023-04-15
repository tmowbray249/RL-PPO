import gymnasium as gym
from Agent import PPOAgent

class Env:

    def __init__(self, env, render=False):
        self.IMPLEMENTED_ENVS = {
            "cp-v0": "CartPole-v0",
            "cp-v1": "CartPole-v1"
        }

        self.env_name = self.set_env(env)
        self.env = gym.make(self.env_name, render_mode="human" if render else "rgb_array")
        self.obs_dims = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.epochs = 0
        self.steps_per_epoch = 4000
        self.train_policy_iterations = 80
        self.train_value_iterations = 80
        self.target_kl = 0.01
        self.agent = PPOAgent(self.obs_dims, self.num_actions, self.steps_per_epoch)


    def set_env(self, env):
        if env in self.IMPLEMENTED_ENVS:
            return self.IMPLEMENTED_ENVS[env] 
        else:
            print(f"Env: {env} not implemented")
            exit(0)

    def train(self, epochs):
        self.epochs = epochs
        self.train_agent()
        self.agent.save_agent(self.env_name, epochs)

    def test(self, epochs):
        self.epochs = epochs
        self.agent.network.load_models(self.env_name, epochs)
        self.test_agent()

    def test_agent(self, num_episodes=5, render=True):
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                self.env.render()

                obs = obs.reshape(1, -1)
                _, action = self.agent.sample_action(obs)
                obs, reward, done, _, _= self.env.step(action[0].numpy())
                total_reward += reward

            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
        
        self.env.close()

    def train_agent(self):
        # Initialize the observation, episode return and episode length
        observation, _, = self.env.reset()
        episode_return, episode_length = 0, 0
        # Iterate over the number of epochs
        for epoch in range(self.epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            num_episodes = 0

            # Iterate over the steps of each epoch
            for t in range(self.steps_per_epoch):

                # Get the logits, action, and take one step in the environment
                observation = observation.reshape(1, -1)
                logits, action = self.agent.sample_action(observation)
                observation_new, reward, done, _, _ = self.env.step(action[0].numpy())
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                value_t = self.agent.network.critic(observation)
                logprobability_t = self.agent.logprobabilities(self.num_actions, logits, action)

                # Store obs, act, rew, v_t, logp_pi_t
                self.agent.memory.store(observation, action, reward, value_t, logprobability_t)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                terminal = done
                if terminal or (t == self.steps_per_epoch - 1):
                    last_value = 0 if done else self.agent.network.critic(observation.reshape(1, -1))
                    self.agent.memory.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, _, = self.env.reset()
                    episode_return, episode_length = 0, 0

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.agent.memory.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self.agent.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for _ in range(self.train_value_iterations):
                self.agent.train_value_function(observation_buffer, return_buffer)

            # Print mean return and length for each epoch
            print(f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}")
        
        self.env.close()
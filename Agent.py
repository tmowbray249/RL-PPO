import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from Memory import PPOMemory
from Models import ActorCritic


class PPOAgent:

    def __init__(self, obs_dims, num_actions, steps_per_epoch):
        # Hyperparameters of the PPO algorithm
        actor_lr = 3e-4
        critic_lr = 1e-3
        self.num_actions = num_actions
        self.policy_optimizer = Adam(learning_rate=actor_lr)
        self.value_optimizer = Adam(learning_rate=critic_lr)
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.network = ActorCritic(obs_dims, num_actions)

        # Initialize the buffer
        self.memory = PPOMemory(obs_dims, steps_per_epoch, self.gamma, self.lam)

    def logprobabilities(self, num_actions, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
        )
        return logprobability
    
    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.network.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action
    
    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
        self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.num_actions, self.network.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.network.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.network.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.num_actions, self.network.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.network.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.network.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.network.critic.trainable_variables))

    def save_agent(self, env_name, epochs):
        self.network.save_models(env_name, epochs)

    def load_agent(self, env_name, epochs):
        self.network.load_models(env_name, epochs)

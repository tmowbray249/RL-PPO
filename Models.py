import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf

class ActorCritic:

    def __init__(self, obs_dims, num_actions):
        # Initialize the actor and the critic as keras models
        hidden_sizes = (64, 64)
        print(obs_dims)
        print(num_actions)
        obs_input = keras.Input(shape=(obs_dims,), dtype=tf.float32)
        print(obs_input)
        
        
        actions = self.build(obs_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=obs_input, outputs=actions)
        
        value = tf.squeeze(
            self.build(obs_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=obs_input, outputs=value)

    def build(self, x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = Dense(units=size, activation=activation)(x)
        return Dense(units=sizes[-1], activation=output_activation)(x)
    
    def save_models(self, env_name, epochs):
        self.actor.save(f"models/{env_name}_actor_epoch_{epochs}.h5")
        self.critic.save(f"models/{env_name}_critic_epoch_{epochs}.h5")


    def load_models(self, env_name, epochs):
        self.actor = keras.models.load_model(f"models/{env_name}_actor_epoch_{epochs}.h5")
        self.critic = keras.models.load_model(f"models/{env_name}_critic_epoch_{epochs}.h5")


# class ActorNetwork(keras.Model):
#     def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
#         super(ActorNetwork, self).__init__()

#         self.fc1 = Dense(fc1_dims, activation='relu')
#         self.fc2 = Dense(fc2_dims, activation='relu')
#         self.fc3 = Dense(n_actions, activation='softmax')

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return x


# class CriticNetwork(keras.Model):
#     def __init__(self, fc1_dims=64, fc2_dims=64):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = Dense(fc1_dims, activation='relu')
#         self.fc2 = Dense(fc2_dims, activation='relu')
#         self.q = Dense(1, activation=None)

#     def call(self, state):
#         x = self.fc1(state)
#         x = self.fc2(x)
#         q = self.q(x)

#         return q
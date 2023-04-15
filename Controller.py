import os
from Env import Env

class Controller:

    def __init__(self, env_name, mode, epochs, render):
        self.env_name = env_name
        self.mode = mode
        self.epochs = epochs
        self.render = render

        # Create a folder for the models
        if not os.path.exists("models"):
            os.makedirs("models")

    def check_inputs(self):
        if type(self.env_name) != str:
            print(f"Invalid env code: {self.env_name}. Must be a string")
            exit()

        try:
            self.epochs = int(self.epochs)
            if self.epochs < 1:
                print(f"Invalid epochs: {self.epochs}. Value should be a number larger than 0")
                exit(0)
        except ValueError:
            print(f"Invalid epochs: {self.epochs}. Value should be a number")
            exit(0)

    def build_env(self):
        return Env(self.env_name, True if self.render == "y" else False)

    def run(self):
        self.check_inputs()
        env = self.build_env()
        
        if self.mode == "test":
            env.test(self.epochs)
        else:
            env.train(self.epochs)
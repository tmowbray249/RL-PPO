from Controller import Controller
import click

@click.command()
@click.option('--env-name', prompt='Enter environment code')
@click.option('--mode', prompt='Test or Train?')
@click.option('--epochs', prompt='How many epochs?')
@click.option('--render', prompt='Render output?(y/n)')
def start_up(env_name, mode, epochs, render):
    controller = Controller(env_name, mode, epochs, render)
    controller.run()
    
if __name__ == '__main__':
    start_up()




        # todo set up plots etc 

        # todo more out to to the console during training 

        # todo comb through all the code to understand it 
            # todo change some of the names to ones i know to make it sink in 

        # todo try with different envs
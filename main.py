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

        # todo try with different envs

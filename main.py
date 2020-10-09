
from agent import PIDControlAgent
from env import CartPoleEnv
from solver import CartPoleSolver

def pid_control_solver():

    env = CartPoleEnv.make()
    # model_maker = DQNMaker()

    # NOTE: kp, ki, kd are tuned manually, they are not the optimal parameter
    # for this PID controller
    agent = PIDControlAgent(env.action_space,
                            env.metadata['video.frames_per_second'],
                            kp=1, ki=0, kd=75)

    # NOTE: pretraining and training stage is not required for this solver
    solver = CartPoleSolver(env=env, agent=agent,
                            skip_pretraining=True,
                            skip_training=True)
    solver.run()







if __name__ == '__main__':
    pid_control_solver()
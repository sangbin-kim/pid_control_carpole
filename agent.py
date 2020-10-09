import gym.spaces

class BasicAgent(object):
    def __init__(self, action_space):
        """
        Parameters
        ----------
        action_space : gym.spaces
            Determine the valid action that agent can generate.
        """
        self.action_space = action_space

    def initialize(self, state):
        pass

    def pretraining_act(self, state):
        """
        Generate action in the stage of pre-training.
        (e.g. initializing data for training).
        """
        return self.action_space.sample()

    def training_act(self, state):
        """
        Generate action in the stage of training model.
        """
        return self.action_space.sample()

    def solving_act(self, state):
        """
        Generate action in the stage of testing model.
        """
        return self.action_space.sample()

    def pretraining_react(self, state, reward):
        """
        React when receiving observation in the stage of pre-training.
        """
        pass

    def training_react(self, state, reward):
        """
        React when receiving observation in the stage of training model.
        """
        pass

    def solving_react(self, state, reward):
        """
        React when receiving observation in the stage of testing model.
        """
        pass



class PIDControlAgent(BasicAgent):
    """
    This agent is a pure PID controller, so its parameters (kp, ki, kd) is not
    going to be tuned automatically.
    However, we can apply a learning model to tune them later.
    """
    def __init__(self, action_space, fs, kp=1.2, ki=1.0, kd=0.001, set_angle=0.0):
        """
        Parameters
        ----------
        action_space : gym.spaces
            Determine the valid action that agent can generate.
        fs : float
            Samping frequency. (Hz)
        kp : float
            Gain of propotional controller.
        ki : float
            Gain of integral controller.
        kd : float
            Gain of derivative controller.
        """
        super(PIDControlAgent, self).__init__(action_space)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.set_angle = set_angle
        self.tau = 1.0/fs

        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

        # cache
        self.output = 0.0
        self.err_prev = 0.0

    def update(self, v_in, v_fb):
        """
        Parameters
        ----------
        v_in : int or float
            Input command.
        v_fb : int or float
            Feedback from observer.

        Returns
        -------
        output : float
            Output command.

        Note
        ----
        Output of PID controller:
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        err = v_in - v_fb

        self.p_term = err
        self.i_term += err*self.tau
        self.d_term = (err - self.err_prev)*self.tau
        self.output = self.kp*self.p_term + self.ki*self.i_term + self.kd*self.d_term

        # update cache
        self.err_prev = err

        return self.output

    def choose_action(self, val):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = 0 if val >= 0 else 1
        elif isinstance(self.action_space, gym.spaces.Box):
            action = None   # rewrite this for continous action space
        return action

    def solving_act(self, state):
        output = self.update(self.set_angle, state[2])
        temp = self.choose_action(output)
        self.action = temp
        return self.action
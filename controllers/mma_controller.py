import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp, kp=8.0, kd=20.0):
        # Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [
            ManiuplatorModel(Tp, 0.1, 0.05),
            ManiuplatorModel(Tp, 0.01, 0.01),
            ManiuplatorModel(Tp, 1., 0.3)
        ]
        self.kp = kp
        self.kd = kd
        self.Tp = Tp
        self.i = 0
        self.last_x = np.zeros(4)
        self.last_u = np.zeros(2)

    def choose_model(self, x):
        # Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        def est_x(model, x, u):
            q_dot = np.reshape(x[2:], (2, 1))
            u = np.reshape(u, (2, 1))
            q_ddot = np.linalg.inv(model.M(x)) @ (u - model.C(x) @ q_dot)
            x_dot = np.concatenate([q_dot, q_ddot])
            return x + self.Tp * x_dot
        
        es = np.zeros(3)
        for i, model in enumerate(self.models):
            e = x - est_x(model, self.last_x, self.last_u)
            es[i] = np.linalg.norm(e)

        self.i = np.argmin(es)
        # print(self.i)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        v = q_r_ddot + q_r_ddot + self.kp * (q_r - q) + self.kd * (q_r_dot - q_dot)

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        
        self.last_x = x
        self.last_u = u
        return u
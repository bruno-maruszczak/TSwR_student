import numpy as np


class ManiuplatorModel: # Xd
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.d1 = self.l1 * 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.d2 = self.l2 * 0.5
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.1
        self.r3 = 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.alfa = self.m1*self.d1**2 + self.I_1 + self.m2*(self.l1**2 + self.d2**2) + self.I_2 + self.m3*(self.l1**2 + self.l2**2) + self.I_3
        self.beta = self.l1*(self.m2*self.d2 + self.m3*self.l2)
        self.ceta = self.m2*self.d2**2 + self.I_2 + self.m3*self.l2**2 + self.I_3

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        m11 = self.alfa + 2*self.beta * np.cos(q2)
        m12 = self.ceta + self.beta * np.cos(q2)
        m21 = self.ceta + self.beta * np.cos(q2)
        m22 = self.ceta

        return np.array([[m11, m12], 
                        [m21, m22]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        c11 = -self.beta * np.sin(q2) * q2_dot
        c12 = -self.beta * np.sin(q2) * (q1_dot + q2_dot)
        c21 = self.beta * np.sin(q2) * q1_dot
        c22 = 0

        return np.array([[c11, c12],
                        [c21, c22]]) 
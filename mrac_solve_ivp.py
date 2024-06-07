import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp
import time
import matplotlib.pyplot as plt

# define tuning params, using ones from example
Gamma_x = np.diag([100, 100])  #learning rate for Kx_hat
Gamma_r = 100  #learning rate for Kr_hat
Gamma_th = np.diag([100, 100, 100])  #learning rate for Theta_hat
Q = np.diag([1, 10])  # Vdot = -e.T @ Q @ e for lyapunov stability on tracking.

# define systems and parameters globally
# real parameters of the system
theta1 = -.018
theta2 = .015
theta3 = -.062
theta4 = .009
theta5 = .021
theta6 = .75

#state space formulation
A = np.array([[0, 1], [theta1, theta2]])
B = np.array([0, 1]).T

#desired second order system params, book has 1 and 0.7 for cmd in book
wn = 1
zeta = .7

# state space reference system
A_ref = np.array([[0, 1], [-wn**2, -2 * zeta * wn]])
B_ref = np.array([0, wn**2])

# solution to algebraic lyapunov equation, but pass in A.T and -Q to fit scipy docs
P = sp.linalg.solve_continuous_lyapunov(A_ref.T, -Q)
print('P (should be PD and symmetric): \n', P)

#ideal gains for testing
ideal_adapt = np.array([-1.3093, -1.8867, 1.3333, -0.062, .009, .021])


def calc_plant_derivs(state, input):
    """
  Note that these actual values are generally unknown, but exist in the plant.
  We assume they are known here for simulation. On hardware, the estimates of these parameters are bounded
  and will converge to something, not necessarily the true plant parameters

  state = [varphi, p].T
  input = delta_a (differential aileron (rad))
  """

    #make sure things are the right sizes
    assert (state.shape == (2, ))
    assert (np.isscalar(input))

    varphi = state[0]
    p = state[1]

    # print('x: ', np.degrees(state))
    # print('u: ', np.degrees(input), '\n')

    # time.sleep(0.1)

    #calculate xdot
    xdot = A @ state + B * theta6 * (input + -.0827 * np.abs(varphi) * p +
                                     .012 * np.abs(p) * p + .028 * varphi**3)

    #rate constraint
    #   max = np.array([0.1,0.1])
    #   xdot = np.clip(xdot, -max, max)

    return xdot


def calc_ref_derivs(ref_state, input):
    '''
  ref_state = [varphi_ref, p_ref].T
  input = varphi_cmd
  '''
    #make sure things are the right sizes
    assert (ref_state.shape == (2, ))
    assert (np.isscalar(input))

    ref_dot = A_ref @ ref_state + B_ref * input

    return ref_dot


def calc_adapt_derivs(adapt_state, input):
    '''
  adapt state = [K_x_hat, K_r_hat, theta_hat].T
  input = [e, x, r], python list, error is a vector, x is a vector, r is a scalar
  '''

    e = input[0]
    x = input[1]
    r = input[2]

    # print('e', e)
    # print('x', x)
    #   print('r', r)

    #make sure things are the right sizes
    assert (adapt_state.shape == (2 + 1 + 3, ))
    assert (e.shape == (2, ))
    assert (x.shape == (2, ))
    assert (np.isscalar(r))

    # note reshaping so that x@x.T returns matrix, not a scalar
    K_x_hat_dot = -Gamma_x @ x.reshape(-1, 1) @ e.reshape(
        -1, 1).T @ P @ B  # 2x2 2x1 1x2 2x2 2x1 = 2x1
    K_r_hat_dot = -Gamma_r * r * e.reshape(-1,
                                           1).T @ P @ B  # 1x2 2x2 2x1 = scalar
    Theta_hat_dot = Gamma_th @ calc_Phi(x).reshape(-1, 1) @ e.reshape(
        -1, 1).T @ P @ B  # 3x3 3x1 1x2 2x2 2x1 = 3x1

    return np.concatenate((K_x_hat_dot, K_r_hat_dot, Theta_hat_dot))


def calc_all_derivs(t, augmented_state):
    plant_state = augmented_state[:2]  #2
    ref_state = augmented_state[2:4]  #2
    # adapt_state = augmented_state[4:] #6

    adapt_state = np.array([
        -(wn**2 + theta1) / theta6, -(2 * zeta * wn + theta2) / theta6,
        wn**2 / theta6, theta3 / theta6, theta4 / theta6, theta5 / theta6
    ])

    freq = 1 / 30
    r = np.radians(15) * (np.sin(2 * np.pi * freq * t) + np.sin(
        2 * np.pi * freq / 2 * t) + np.sin(2 * np.pi * freq * 2 * t))
    u = calc_control_input(adapt_state, r, plant_state)
    e = plant_state - ref_state

    plant_dot = calc_plant_derivs(plant_state, u)
    ref_dot = calc_ref_derivs(ref_state, r)
    adapt_input = [e, plant_state, r]
    adapt_dot = calc_adapt_derivs(adapt_state, adapt_input) * 0

    derivs = np.hstack([plant_dot, ref_dot, adapt_dot])

    return derivs


def calc_control_input(adapt, ref, state):
    Kx_hat = adapt[:2]
    Kr_hat = adapt[2]
    Theta_hat = adapt[3:]

    # 1x2 2x1 + 1x1 1x1 - 1x3 3x1 = 1x1
    #   u = Kx_hat.T @ state + Kr_hat.T * ref - Theta_hat.T @ calc_Phi(state)

    #enforce model matching condition in this case. Will need to comment out later.
    assert (np.allclose(A + B.reshape(-1,1) @ np.array([[theta6]]) @ Kx_hat.reshape(-1,1).T, A_ref))
    assert (np.allclose(B.reshape(-1,1) @ np.array([[theta6]]) @ Kr_hat.reshape(-1,1), B_ref.reshape(-1,1)))

    u = Kx_hat.T @ state + Kr_hat.T * ref - (1 / theta6) * np.array(
        [theta3, theta4, theta5]).T @ calc_Phi(state)

    return u


def calc_Phi(state):
    #make sure things are the right sizes
    assert (state.shape == (2, ))

    varphi = state[0]
    p = state[1]

    Phi = np.array([np.abs(varphi) * p, np.abs(p) * p, varphi**3])

    return Phi


# simulation times
dt = .01
tspan = np.arange(0, 60 * 5, dt)
sol = solve_ivp(calc_all_derivs, (tspan[0], tspan[-1]),
                t_eval=tspan,
                y0=np.zeros(10),
                max_step=.01)

#plot the sol from solve_ivp
tracking_error = sol.y[0] - sol.y[2]
plt.figure()
plt.plot(tspan, np.degrees(tracking_error))
plt.grid(True)

fig, axs = plt.subplots(2, 1)
axs[0].plot(tspan, np.degrees(sol.y[0]), label='varphi')
axs[0].plot(tspan, np.degrees(sol.y[2]), '--', label='varphi_ref')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tspan, np.degrees(sol.y[1]), label='p')
axs[1].plot(tspan, np.degrees(sol.y[3]), '--', label='p_ref')
axs[1].legend()
axs[1].grid(True)

plt.show()

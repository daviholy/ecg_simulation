import numpy.typing as npt
import numpy as np


def _dif_x13(x1_3, x2_4, C):
  return x1_3 - x2_4 - C*x1_3*x2_4 - x1_3*x2_4**2

def _dif_x2(x1, x2, x4, C, H, beta):
  return H*x1 - 3*x2 + C*x1*x2 + x1*x2**2 + beta*(x4 - x2)

def _dif_x4(x1, x2, x3, x4, C, H, beta):
  return H*x3 - 3*x4 + C*x3*x4 + x3*x4**2 + 2*beta*(x2 - x4)

def _runge_kutta_4(x, beta, H, C, step, gamma) :
    kx = np.zeros((4,4))

    # First iteration is a special case
    kx[0,0], kx[1,0] = _dif_x13(x[0], x[1], C), _dif_x2(x[0], x[1], x[3], C, H, beta)
    kx[2,0], kx[3,0] = _dif_x13(x[2], x[3], C), _dif_x4(x[0], x[1], x[2], x[3], C, H, beta)

    new_xs = [0] * 4
    for i in range(1, 4):
        factor = 0.5 if i < 3 else 1
        for j,_ in enumerate(new_xs):
          new_xs[j] = x[j] + factor * step * kx[j, i - 1]

        kx[0, i] = _dif_x13(new_xs[0], new_xs[1], C)
        kx[1, i] = _dif_x2(new_xs[0], new_xs[1], new_xs[3], C, H, beta)
        kx[2, i] = _dif_x13(new_xs[2], new_xs[3], C)
        kx[3, i] = _dif_x4(new_xs[0], new_xs[1], new_xs[2], new_xs[3], C, H, beta)

    # Final computation
    for i,_ in enumerate(new_xs):
      new_xs[i] = x[i] + step / 6 * (kx[i,0] + 2 * kx[i,1] + 2 * kx[i,2] + kx[i,3]) * gamma

    return new_xs

def generate_ecg(C: float, H: float, alpha: npt.NDArray, beta: float, HRbpm: int, time: float, step: float, startup_time: float):
  time += startup_time
  X = np.zeros((4, int(time/step) + 1))
  X[2, 0] = 0.1
  gamma = 0.08804 * HRbpm - 0.06754

  for t in range(int(time/step)):
    new_xs = _runge_kutta_4(X[:,t], beta, H, C, step, gamma) 

    for i, xi in enumerate(new_xs):
      X[i, t+1] = xi

  ECG = alpha[0] * X[0,:] + alpha[1] * X[1, :] + alpha[2] * X[2, :] + alpha[3] * X[3, :]
  return ECG[int(startup_time / step) + 1:], X[int(startup_time/step) + 1 :]
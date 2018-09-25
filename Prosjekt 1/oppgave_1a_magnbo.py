from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from random import seed, random
from sklearn.utils import resample
from sklearn.metrics import r2_score

X1 = np.arange(0, 1, 0.05)
Y1 = np.arange(0, 1, 0.05)
X1, Y1 = np.meshgrid(X1, Y1)


def franke_function(x_0, y_0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x_0 - 2)**2) - 0.25 * ((9 * y_0 - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x_0 + 1)**2) / 49.0 - 0.1 * (9 * y_0 + 1))
    term3 = 0.5 * np.exp(-(9 * x_0 - 7)**2 / 4.0 - 0.25 * ((9 * y_0 - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x_0 - 4)**2 - (9 * y_0 - 7)**2)
    return term1 + term2 + term3 + term4


Z1 = franke_function(X1, Y1)

np.random.seed(2)

Z2 = Z1.flatten()
X2 = X1.flatten()
Y2 = Y1.flatten()
C = np.ones((400, 1))
XB1 = np.c_[C, X2, Y2]
XB2 = np.c_[C, X2, Y2, X2**2, Y2**2, X2 * Y2]
XB3 = np.c_[C, X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2]
XB4 = np.c_[C, X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2, X2**4, X2**3 * Y2,
            X2**2 * Y2**2, X2 * Y2**3, Y2**4]
XB5 = np.c_[C, X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2, X2**4, X2**3 * Y2,
            X2**2 * Y2**2, X2 * Y2**3, Y2**4, X2**5, X2**4 * Y2, X2**3 * Y2**2, X2**2 * Y2**3
            , X2 * Y2**4, Y2**5]

# z_noise = z1 + 0.1 * np.random.randn(400, 1)
def ridge(xb, lam_0):
    m, n = xb.shape
    beta = (np.linalg.inv(xb.T @ xb + lam_0 * np.identity(n)).dot(xb.T).dot(Z2))
    xbnew = xb
    zpredict = xbnew @ beta
    zpredict2 = np.reshape(zpredict, (20, 20))
    variance = 0
    for i in range(400):
        variance += (1/(400-n-1)) * (Z2[i] - zpredict[i])**2
    covariance_matrix = np.linalg.inv(xb.T @ xb) * variance
    mse = 0.
    for i in range(400):
        mse += (1/400) * (Z2[i] - zpredict[i])**2
    u = 0.
    b = 0.
    zmean = np.mean(Z1)
    for i in range(400):
        u += (Z2[i] - zpredict[i])**2
        b += (Z2[i] - zmean)**2
    rr = 1 - u/b
    rr_boot = np.zeros(30)
    mse_boot = np.zeros(30)

    for i in range(30):
        xbnew_boot_train, z_boot_train = resample(xbnew, Z2, n_samples=100)
        oob_xb = np.array([x for x in xbnew if x.tolist() not in xbnew_boot_train.tolist()])
        oob_z = np.array([x for x in Z2 if x not in z_boot_train])
        xbnew_boot_test, z_boot_test = resample(oob_xb ,oob_z , n_samples=100)
        beta_boot_train = (np.linalg.inv(xbnew_boot_train.T @ xbnew_boot_train + lam_0 *
                                   np.identity(n)).dot(xbnew_boot_train.T).dot(z_boot_train))
        zpredict_boot_test = xbnew_boot_test @ beta_boot_train
        for j in range(100):
            mse_boot[i] += (1/100) * (z_boot_test[j] - zpredict_boot_test[j])**2
        u = 0
        b = 0
        rr_boot[i] = r2_score(z_boot_test, zpredict_boot_test)
    mse_boot_mean = np.mean(mse_boot)
    mse_boot_2std = 2 * np.std(mse_boot)
    rr_boot_mean = np.mean(rr_boot)
    rr_boot_2std = 2* np.std(rr_boot)
    beta_confidence = 2 * np.linalg.norm(np.diag(covariance_matrix))
    return (beta, beta_confidence, mse, rr, mse_boot_mean, mse_boot_2std, rr_boot_mean,
            rr_boot_2std)

BETA, BETA_CONFIDENCE, MSE, RR, MSE_BOOT_MEAN, MSE_BOOT_2STD, RR_BOOT_MEAN, RR_BOOT_2STD = ridge(XB1,
                                                                                                 0)
print(BETA_CONFIDENCE)
# FIG = plt.figure()
# AX = FIG.gca(projection='3d')
# SURF = AX.plot_surface(X1, Y1, Z1, cmap=cm.viridis, linewidth=0)
# FIG = plt.figure()
# AX = FIG.gca(projection='3d')
# AX.plot_surface(X1, Y1, ZPREDICT2, cmap=cm.viridis, linewidth=0)
# plt.show()

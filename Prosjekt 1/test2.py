# Singular value decomposition inverse
xbdot = xb.T.dot(xb)
U, s, VT = svd(xbdot)
d = 1.0 / s
D = np.zeros(xbdot.shape)
D[:xbdot.shape[1], :xbdot.shape[1]] = diag(d)
inv = VT.T.dot(D.T).dot(U.T)
#beta = inv.dot(xb.T).dot(z)



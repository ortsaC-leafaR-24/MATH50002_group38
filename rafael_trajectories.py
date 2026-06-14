import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import linregress
import scipy.interpolate as interpolate
import porespy as ps
from itertools import accumulate


class NoStrangeAttractorError(Exception):
    pass

def ortho_offset(x0, _dir, dist):
    """return x0 + e_(_dir)*dist for e_k is the (n-1)st elementary basis vector"""
    return x0 + dist*np.eye(1, x0.shape[0], _dir)[0]

class DifferenceFunction:
    """
    Class modelling difference functions.

    Attributes
    ––––––––––
    func: function
    The function mapping (x_n, y_n, ...) to (x_(n+1), y_(n+1), ...).
    We assume output is as numpy array.

    dim: Int
    Dimension of input/output of func.
    """

    def __init__(self, func, dim, *args):
        """initialise a new DifferenceFunction"""
        self.func = func
        self.dim = dim
        if args:
            self.jac = args[0]
        else:
            self.jac = None

    def __call__(self, _in):
        """call self on DifferenceFunction"""
        if not hasattr(_in, "__array__"):
            raise TypeError(f"{_in} is {type(_in)} and not array-like.")
        elif _in.shape != (self.dim,):
            raise ValueError(f"{_in} is shape {_in.shape}, not {(self.dim,)}")
        return self.func(*_in) 
    
    def run_difference(self, x0, n=200):
        """run difference function f on x_0 n times"""
        _out = [x0]
        for p in range(n):
            _out.append(self(_out[p]))
        return np.array(_out)


    def jacobian(self, x0, tol = 1e-7):
        """ return approximation for Jacobian at location x0.""" 
        if self.jac:
            return self.jac(*x0)
        else:
            return np.array([
                            [-(self(x0)[k]-self(ortho_offset(x0, j, tol))[k])/tol
                              for j in range(self.dim)] for k in range(self.dim)])

    def lyapunov(self, x0, u0, niter = 100, tol = 1e-7):
        """return Lyapunov exponent in direction u0. u0 is assumed to 
        have norm 1"""
        _alpha = self.run_difference(x0, niter)
        big_dm = np.identity(self.dim)
        for j in range(niter):
            big_dm = self.jacobian(_alpha[j,:]) @ big_dm
            
        return 1/niter*np.log(np.linalg.norm(big_dm @ u0))

    def stability_test(self, x0, niter=20, tol=0):
        """Determine whether the DifferenceFunction is stable near x0"""
        lyap_exps = [self.lyapunov(x0, np.eye(1, self.dim, j)[0], niter, tol) 
                     for j in range(self.dim)]
        return (lyap_exps < np.zeros(self.dim)).all()
            

    

def quad_confirm_attr_multi_d(A_, x_0=(0.05, 0.05), maxtol = 1e7, mintol=1e-7, num=125000):
    """Confirm a given 2d quadratic equation has a strange attractor."""
    n_ = len(A_[0])
    x0 = [np.array([*x_0])]
    func_x = lambda x, y: sum([sum([A_[0, j, k]*x**j*y**k for j in range(n_)]) for k in range(n_)])
    func_y = lambda x, y: sum([sum([A_[1, j, k]*x**j*y**k for j in range(n_)]) for k in range(n_)])
    func = lambda x, y: np.array([func_x(x, y), func_y(x, y)])
    func_a_x = lambda a, x, y: sum([sum([j*A_[a, j, k]*x**(j-1)*y**k for j in range(1, n_)]) for k in range(n_)])
    func_a_y = lambda a, x, y: sum([sum([k*A_[a, j, k]*x**j*y**(k-1) for j in range(n_)]) for k in range(1, n_)])
    dfunc = lambda x, y: np.array([[func_a_x(0, x, y), func_a_y(0, x, y)], 
                                   [func_a_x(1, x, y), func_a_y(1, x, y)]])
    quadratic_func = DifferenceFunction(func, 2, dfunc)
    l_sum = 0
    while len(x0) < num:
        lyap_exp = np.abs(np.linalg.det(quadratic_func.jacobian(x0[-1])))
        if lyap_exp > 0:
            l_sum += np.log(lyap_exp)
            L = l_sum/(len(x0))
        if np.linalg.norm(x0[-1]) > maxtol:
            raise NoStrangeAttractorError(f"Unbounded: norm(func{x0[-1]}) > {maxtol}")
        if len(x0) > 2 and np.linalg.norm(x0[-1] - x0[-2]) < mintol:
            raise NoStrangeAttractorError(f"Fixed point: |{x0[-1]} - {x0[-2]}| < {mintol}")
        if len(x0) >= 1000 and L < 0:
            raise NoStrangeAttractorError("Limit cycle: Lyapunov exponent < 0")
        x0.append(quadratic_func(x0[-1]))
    return np.array(x0[999:]), quadratic_func, A_

def quad_attr_search_multi_d(a_0 = np.array([
        [[0, 0.1, 0.2], [0.1, 0.2, 0], [0.2, 0, 0]],
        [[0, 0.1, 0.2], [0.1, 0.2, 0], [0.2, 0, 0]]
        ])):
    """Search for strange attractor given shape of parameters"""
    n_ = len(a_0[0])
    while True:
        try:
            return quad_confirm_attr_multi_d(a_0)
        except NoStrangeAttractorError:
            a_0 =np.array([[[*[(random.random()-0.5)*6 for j in range(n_-k)],
                    *[0 for j in range(k)]] for k in range(n_)]
                 for p in range(2)])

def attr_plot_multi_d(title, a_0 = None):
    """Find and plot a multi-dimensional strange attractor"""
    x0_data, strange_system, vals = quad_attr_search_multi_d(a_0)
    plt.scatter(x0_data[:,0], x0_data[:, 1], marker = ".", s = 0.002, alpha=0.6)
    plt.title(title)
    plt.show()
    return strange_system, x0_data, vals


def splinedim(points, u = None, *args, **kwargs):
    """
    Generate spline approximation of a curve through points
    at evenly-spaced time intervals.
    """
    if u is None: 
        u = list(range(len(points)))
    #u_ is just u so we separate it out of the output with this
    _out, u_ = interpolate.make_splprep(points.transpose(), u=u, *args, **kwargs)
    return _out

def full_dim_analysis(info, n=10000):
    """Find box-counting dimension of a trajectory."""
    sf = max(max(info[:,0])-min(info[:,0]), max(info[:,1])-min(info[:,1]))
    dict_info = {"X" : info[-n:, 0]*5e3/sf, 
                 "Y" : info[-n:, 1]*5e3/sf, 
                 "Z" : np.zeros([len(info[-n:, 0])])
                 }
    regressinfo = ps.metrics.boxcount(ps.generators.spheres_from_coords(dict_info), bins=10)
    return linregress(np.log(regressinfo.size), -np.log(regressinfo.count))

attractors_ = []
[dims_x, dims_dxdt_1, dims_dxdt_2, 
 dims_x_int_1, dims_x_int_2] = [[] for j in range(5)]
for j in range(60):
    attractor, info, a_vals = attr_plot_multi_d(title = f"Attractor {j+1}", 
                                            a_0 = np.array([
        [[0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0]],
        [[0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0]]
        ]))
    num_ = len(info[:,0])
    #centre attractor at 0 for purposes of integration
    info = info - np.array([[np.mean(info[:,0]) for j in range(num_)],
                           [np.mean(info[:,1]) for j in range(num_)]]).transpose()
    attractors_.append(attractor)
    
    spline_info = splinedim(info)
    spline_info_diff, spline_info_int = [spline_info.derivative(), 
                                         spline_info.antiderivative()]
    

    dx_dt_1 = spline_info_diff(np.array(range(num_))).transpose()
    dx_dt_2 = np.array([info[k+1, :] - info[k, :] for k in range(num_-1)])
    x_int_1 = spline_info_int(np.array(range(num_))).transpose()
    x_int_2 = np.array(list(accumulate([info[k,:] for k in range(num_)])))
    
    alpha = full_dim_analysis(info)
    
    print(f"Attractor {j+1}:")
    
    x_str = " + ".join([" + ".join([f'({a_vals[0, j, k]})*x^{j}*y^{k}' for j in range(3-k)]) 
                        for k in range(3)])
    y_str = " + ".join([" + ".join([f'({a_vals[1, j, k]})*x^{j}*y^{k}' for j in range(3-k)]) 
                        for k in range(3)])
    print(f""""Formula:
f(x, y) = ({x_str}, {y_str})""")
    
    print(f'Box counting dimension: {alpha[0]} \n \n r^2 : {alpha[2]**2} \n \n')
    #i could probably turn this into some test tos ee if they are the same or something
    #maybe another day who knows
    
    plt.scatter(dx_dt_1[:,0], dx_dt_1[:, 1], marker = ".", s = 0.002, alpha=0.6)
    plt.title("Spline derivative approximation")
    plt.show()
    
    alpha_2 = full_dim_analysis(dx_dt_1)
    print('Spline derivative approximation:')
    print(f'Box counting dimension: {alpha_2[0]} \n \n r^2 : {alpha_2[2]**2} \n \n')

    plt.scatter(dx_dt_2[:,0], dx_dt_2[:,1], marker = ".", s = 0.002, alpha=0.6)
    plt.title("Discrete derivative approximation")
    plt.show()

    alpha_3 = full_dim_analysis(dx_dt_2)
    print('Discrete derivative approximation:')
    print(f'Box counting dimension: {alpha_3[0]} \n \n r^2 : {alpha_3[2]**2} \n \n \n')

    plt.scatter(x_int_1[:,0], x_int_1[:,1], marker = ".", s = 0.002, alpha=0.6)
    plt.title("Continuous antiderivative approximation")
    plt.show()

    alpha_4 = full_dim_analysis(x_int_1)
    print('Continuous antiderivative approximation:')
    print(f'Box counting dimension: {alpha_4[0]} \n \n r^2 : {alpha_4[2]**2} \n \n \n')

    plt.scatter(x_int_2[:,0], x_int_2[:,1], marker = ".", s = 0.002, alpha=0.6)
    plt.title("Discrete antiderivative approximation")
    plt.show()

    alpha_5 = full_dim_analysis(x_int_2)
    print('Discrete antiderivative approximation:')
    print(f'Box counting dimension: {alpha_5[0]} \n \n r^2 : {alpha_5[2]**2} \n \n \n')

    dims_x.append(alpha[0])
    dims_dxdt_1.append(alpha_2[0])
    dims_dxdt_2.append(alpha_3[0])
    dims_x_int_1.append(alpha_4[0])
    dims_x_int_2.append(alpha_5[0])

print("Plotting dimensions against each other:")

plt.scatter(dims_x, dims_dxdt_1, c="b", s=0.5)
plt.scatter(np.linspace(1, 2, 200), np.linspace(1, 2, 200), c="r", s=0.05)
plt.xlabel("Dimension of Attractor")
plt.ylabel("Dimension of Spline Derivative Approximation")
plt.show()
print(linregress(dims_x, dims_dxdt_1), "\n \n \n")

plt.scatter(dims_x, dims_dxdt_2, c="b", s=0.5)
plt.scatter(np.linspace(1, 2, 200), np.linspace(1, 2, 200), c="r", s=0.05)
plt.xlabel("Dimension of Attractor")
plt.ylabel("Dimension of Discrete Derivative Approximation")
plt.show()

print(linregress(dims_x, dims_dxdt_2), "\n \n \n")

plt.scatter(dims_x, dims_x_int_1, c="b", s=0.5)
plt.scatter(np.linspace(1, 2, 200), np.linspace(1, 2, 200), c="r", s=0.05)
plt.xlabel("Dimension of Attractor")
plt.ylabel("Dimension of Spline Antiderivative Approximation")
plt.show()
print(linregress(dims_x, dims_x_int_1), "\n \n \n")

plt.scatter(dims_x, dims_x_int_2, c="b", s=0.5)
plt.scatter(np.linspace(1, 2, 200), np.linspace(1, 2, 200), c="r", s=0.05)
plt.xlabel("Dimension of Attractor")
plt.ylabel("Dimension of Discrete Antiderivative Approximation")
plt.show()
print(linregress(dims_x, dims_x_int_2), "\n \n \n")

plt.scatter(dims_dxdt_1, dims_dxdt_2, s=0.5)
plt.scatter(np.linspace(1, 2, 200), np.linspace(1, 2, 200), c="r", s=0.05)
plt.xlabel("Dimension of Continuous Derivative Approximation")
plt.ylabel("Dimension of Discrete Derivative Approximation")
plt.show()

print(linregress(dims_dxdt_1, dims_dxdt_2), "\n \n \n")

plt.scatter(dims_x_int_1, dims_x_int_2, s=0.5)
plt.scatter(np.linspace(1, 2, 200), np.linspace(1, 2, 200), c="r", s=0.05)
plt.xlabel("Dimension of Continuous Antiderivative Approximation")
plt.ylabel("Dimension of Discrete Antiderivative Approximation")
plt.show()

print(linregress(dims_dxdt_1, dims_dxdt_2), "\n \n \n")

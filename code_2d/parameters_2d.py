import numpy as np      

def parameters(num_types=1):
    if num_types == 1: ##one layer of soil
        #sandy deposit (B. Liu, 2015)
        mu = 1.27 * 10**7 #shear modulus [N/m^2]
        vp = 0.3 #poisson ratio
        E = mu * 2 * (1+vp) #[N/m^2] or [Pa] young's modulus
        labda = (E*vp) / ((1+vp)*(1-2*vp)) #[Pa] lamé's constant
        p = 0.425 #porosity
        P_0 = 10**5#1.52 * 10**5 #[Pa] atmos. pressure 10**5
        s = 0.996 #[m^2/N] saturation grad
        beta_0 = 0.5 * 10**(-9) #[m^2/N] compressibility of pure water
        K_s = 1.8 * 10**(-4) #[m/s] hydraulic conductivity
        
        beta = s*beta_0 + (1.0-s)/P_0 #if 0.0, then incompressible; else, then compressible
        gamma_w = 10**4 #[N/m^3]

        rho_w = 1000 #[kg/m^3] density of water
        g = 10 #[m/s^2] #gravity accelleration
        D = 5.2 #[m] water depth
        
    elif num_types == 2: ##two layers of soil
        #if [a, b], then a is property of medium sand (dense) and b of fine sand (loose)
        d_50 = np.array([0.38,0.12]) #mean size of the grains [mm]
        d_10 = np.array([0.23,0.03]) #effective size of the grains [mm]
        K_s = 0.01 * d_10**2 #hydraulic conductivity m/s
        mu = np.array([3.9 * 10**6,7.7 * 10**6]) #10**(7) #shear modulus [N/m^2] or [Pa]
        vp = np.array([0.27,0.3]) #poisson ratio
        E = mu * 2 * (1+vp) #[N/m^2] or [Pa] young's modulus
        labda = np.zeros(2)
        labda[0] = (E[0]*vp[0]) / ((1+vp[0])*(1-2*vp[0])) ##[N/m^2] or [Pa] lamé's constant
        labda[1] = (E[1]*vp[1]) / ((1+vp[1])*(1-2*vp[1]))
        p = np.array([0.44, 0.4]) #porosity
        
        P_0 = 10**5 #[Pa] atmos. pressure 10**5
        s = 0.996 #[m^2/N] saturation grad
        beta_0 = 0.5 * 10**(-9) #[m^2/N] compressibility of pure water
        K_s = 1.8 * 10**(-4) #[m/s] hydr. cond.
        
        beta = s*beta_0 + (1.0-s)/P_0 #if 0.0, then incompressible; else, then compressible
        gamma_w = 10**4 #[N/m^3]

        rho_w = 1000 #[kg/m^3] density of water
        g = 10 #[m/s^2] #gravity accelleration
        D = 5.2 #[m] water depth

    return beta, K_s, gamma_w, p, mu, labda#, rho_w, g, D


def ini_fem():
    L = 1.0 #lenght (x-direction)
    Z = 2.0 #depth (z-direction)
    m_x = 25#50 #number of subdomains in x-direction
    m_z = 50#100 #number of subdomains in z-direction
    p_x = 1 #degree in x-direction
    p_z = 1 #degree in z-direction
    k = 0   #smoothness
    n_q = 50 #number of integration points
    integrate = True #integration of reference element functions
    return L, Z, m_x, m_z, p_x, p_z, k, n_q, integrate

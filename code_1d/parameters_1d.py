import numpy as np      

def parameters(num_types=1):
    if num_types == 1:
        #sandy deposit (B. Liu, 2015)
        mu = 1.27 * 10**7 #shear modulus [N/m^2]
        vp = 0.3 #poisson's ratio
        E = mu * 2 * (1+vp) #[N/m^2] or [Pa] young's modulus
        labda = (E*vp) / ((1+vp)*(1-2*vp)) #[Pa] lamé's constant
        p = 0.425 #porosity
        K_s = 1.8 * 10**(-4) #[m/s] hydraulic conductivity
        gamma_w = 9810 #[N/m^2] specific weight

        rho_w = 1000 #[kg/m^3] density of water
        g = 9.81 #[m/s^2] #gravity accelleration
        D = 5.2 #[m] water depth
        
    elif num_types == 2:
        #if [a, b], then a is property of medium sand and b of SBM
        d_10 = np.array([0.23,0.3])
        mu = np.array([1.0*10**5, 2.0 * 10**5]) #shear modulus [N/m^2] or [Pa]
        vp = np.array([0.3,0.3]) #poisson's ratio
        E = mu * 2 * (1+vp) #[N/m^2] or [Pa] young's modulus
        labda = np.zeros(2)
        labda[0] = (E[0]*vp[0]) / ((1+vp[0])*(1-2*vp[0])) #[N/m^2] or [Pa] lamé's constant
        labda[1] = (E[1]*vp[1]) / ((1+vp[1])*(1-2*vp[1]))
        p = np.array([0.44, 0.4])
        gamma_w = 10**4 #[N/m^2] specific weight
        K_s = 0.01 * d_10**2 #Hazen hydr. cond. [m/s]

    P_0 = 10**5 #[Pa] atmos. pressure
    s = 1.0 #[m^2/N] saturation grad
    beta_0 = 0.5 * 10**(-9) #[m^2/N] compressibility of pure water  
    beta = s*beta_0 + (1.0-s)/P_0 #if near 0.0, then incompressible; else, then compressibl
    
    return beta, K_s, gamma_w, p, mu, labda

def ini_fem():
    L = 1.8 
    m = 1800
    p = 1 #p=1,k=0 -> hat basis-functions
    k = 0
    n_q = 1000
    integrate=True
    return L, m, p, k, n_q, integrate

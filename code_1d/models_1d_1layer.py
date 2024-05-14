import numpy as np
import copy
import scipy as sp
import parameters_1d
import fem_functions_1d
import plot_functions_1d


def biot_1d(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = 3.0, dt=0.01):    
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_uz = gK * copy.deepcopy(C)
    alpha_P = gK * p * beta * copy.deepcopy(A)

    #Numerical model + neumann boundary conditions
    Mnew_11 = lm*copy.deepcopy(B)
    Mnew_12 = copy.deepcopy(C)
    Mnew_21 = copy.deepcopy(alpha_uz)
    Mnew_22 = copy.deepcopy(alpha_P) + dt*copy.deepcopy(B)
    
    Mold_11 = copy.deepcopy(zero_matrix)
    Mold_12 = copy.deepcopy(zero_matrix)
    Mold_21 = copy.deepcopy(alpha_uz)
    Mold_22 = copy.deepcopy(alpha_P)
    
    #Dirichlet boundary conditions    
    #BC: P = Fzz at z=0 in 2nd eq
    Mnew_21[-1,:] = 0.0
    Mnew_22[-1,:] = 0.0
    Mold_21[-1,:] = 0.0
    Mold_22[-1,:] = 0.0
    Mnew_22[-1,-1] = 1.0

    #BC: uz = 0 at z=-n_z in 2nd eq
    Mnew_11[0,:] = 0.0
    Mnew_12[0,:] = 0.0
    Mold_11[0,:] = 0.0
    Mold_12[0,:] = 0.0
    Mnew_11[0,0] = 1.0
    
    #Make matrices and vectors to solve numerical model
    Mnew_1 = np.hstack((Mnew_11, Mnew_12))
    Mnew_2 = np.hstack((Mnew_21, Mnew_22))
    Mnew = np.vstack((Mnew_1, Mnew_2))
    
    Mold_1 = np.hstack((Mold_11, Mold_12))         
    Mold_2 = np.hstack((Mold_21, Mold_22))
    Mold = np.vstack((Mold_1, Mold_2))

    Mrhs = np.hstack((zero_vector, zero_vector))

    Neps = copy.deepcopy(A) 
    Nuz = copy.deepcopy(C)
    Nrhs = copy.deepcopy(zero_vector)

    #Make matrices sparse
    sM_new = sp.sparse.csr_matrix(Mnew)
    sM_old = sp.sparse.csr_matrix(Mold)
    sN_uz = sp.sparse.csr_matrix(Nuz)
    sN_eps = sp.sparse.csr_matrix(Neps)
    
    #Initialise variables
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    num_t = 2 #amount of timesteps that will be saved
    num_elt = np.shape(B)[1] #number of elements
    coeff = np.zeros((4,num_t,num_elt))
    time = np.zeros(num_t)
    
    k = 0
    #start time for-loop
    for t in np.arange(dt, time_stop+dt, dt):                    
        Mrhs[-1] = fnc(t)
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
    
        if t == (time_stop/10.25)/4 or t == time_stop:#j%num == 0:
            uz = S[0:num_elt]
            P = S[num_elt:]
            eps = sp.sparse.linalg.spsolve(sN_eps, Nrhs + sN_uz.dot(uz))
            eff_stress_zz = -(labda + 2*mu)*eps
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            time[k] = t
            k += 1
    
    return coeff, time

def newS_1d(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = 3.0, dt=0.01):
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_S = gK * (p * beta + 1.0/lm) * copy.deepcopy(A)

    #Numerical model + neumann boundary conditions
    Mnew = copy.deepcopy(alpha_S) + dt*copy.deepcopy(B)
    Mold = copy.deepcopy(alpha_S)

    Neps = copy.deepcopy(A)
    Nuz = copy.deepcopy(C)
    Nrhs = copy.deepcopy(zero_vector)
    
    #Dirichlet boundary conditions   
    #BC: P = Fzz at z=0 in 1st eq
    Mnew[-1,:] = 0.0
    Mold[-1,:] = 0.0
    Mnew[-1,-1] = 1.0

    #BC: u_z = 0 at z=-n_z in 2nd eq
    Neps[0,:] = 0.0
    Nuz[0,:] = 0.0
    Nuz[0, 0] = 1.0
    Nrhs[0] = 0.0

    Mrhs = np.hstack((zero_vector))
    
    #Make matrices sparse
    sM_new = sp.sparse.csr_matrix(Mnew)
    sM_old = sp.sparse.csr_matrix(Mold)
    sN_uz = sp.sparse.csr_matrix(Nuz)
    sN_eps = sp.sparse.csr_matrix(Neps)

    #Initialise variables
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    num_t = 2 #amount of timesteps that will be saved
    num_elt = np.shape(B)[1] #number of elements
    coeff = np.zeros((4,num_t,num_elt))
    time = np.zeros(num_t)
    
    k=0
    #start time for-loop
    for t in np.arange(dt, time_stop+dt, dt):        
        Mrhs[-1] = fnc(t)
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
    
        if t == (time_stop/10.25)/4 or t == time_stop:
            eps = S/lm
            P = S
            uz = sp.sparse.linalg.spsolve(sN_uz, Nrhs + sN_eps.dot(eps))
            eff_stress_zz = -(labda + 2*mu)*eps
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            time[k] = t
            k += 1
    
    return coeff, time

def strict_new_1d(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = 3.0, dt=0.01):
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_eps = gK * copy.deepcopy(A)
    alpha_P = gK * p * beta * copy.deepcopy(A)

    #Numerical model + neumann boundary conditions
    Mnew_11 = copy.deepcopy(alpha_eps) + dt*lm*copy.deepcopy(B)
    Mnew_12 = copy.deepcopy(alpha_P)
    Mnew_21 = copy.deepcopy(alpha_eps)
    Mnew_22 = copy.deepcopy(alpha_P) + dt*copy.deepcopy(B)
    
    Mold_11 = copy.deepcopy(alpha_eps)
    Mold_12 = copy.deepcopy(alpha_P)
    Mold_21 = copy.deepcopy(alpha_eps)
    Mold_22 = copy.deepcopy(alpha_P)

    Neps = copy.deepcopy(A)
    Nuz = copy.deepcopy(C)
    Nrhs = copy.deepcopy(zero_vector)
    
    #Dirichlet boundary conditions   
    #BC: P = Fzz at z=0 in 2nd eq
    Mnew_21[-1,:] = 0.0
    Mnew_22[-1,:] = 0.0
    Mold_21[-1,:] = 0.0
    Mold_22[-1,:] = 0.0
    Mnew_22[-1,-1] = 1.0

    #BC: (labda + 2*mu) eps= P at z=0 ('strict' boundary condition)
    Mnew_11[-1,:] = 0.0
    Mnew_12[-1,:] = 0.0
    Mold_11[-1,:] = 0.0
    Mold_12[-1,:] = 0.0
    Mnew_11[-1,-1] = lm
    Mnew_12[-1,-1] = -1.0

    #BC: u_z = 0 at z=-n_z in 3rd eq
    Neps[0,:] = 0.0
    Nuz[0,:] = 0.0
    Nuz[0, 0] = 1.0
    Nrhs[0] = 0.0
    
    #Make matrices
    Mnew_1 = np.hstack((Mnew_11, Mnew_12))
    Mnew_2 = np.hstack((Mnew_21, Mnew_22))
    Mnew = np.vstack((Mnew_1, Mnew_2))
    
    Mold_1 = np.hstack((Mold_11, Mold_12))
    Mold_2 = np.hstack((Mold_21, Mold_22))
    Mold = np.vstack((Mold_1, Mold_2))

    Mrhs = np.hstack((zero_vector, zero_vector))
    
    #Make matrices sparse
    sM_new = sp.sparse.csr_matrix(Mnew)
    sM_old = sp.sparse.csr_matrix(Mold)
    sN_uz = sp.sparse.csr_matrix(Nuz)
    sN_eps = sp.sparse.csr_matrix(Neps)

    #Initialise variables
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    num_t = 2 #amount of timesteps that will be saved
    num_elt = np.shape(B)[1] #number of elements
    coeff = np.zeros((4,num_t,num_elt))
    time = np.zeros(num_t)
    
    k=0
    #start time for-loop
    for t in np.arange(dt, time_stop+dt, dt):        
        Mrhs[-1] = fnc(t)
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
    
        if t == (time_stop/10.25)/4 or t == time_stop:
            eps = S[0:num_elt]
            P = S[num_elt:]
            uz = sp.sparse.linalg.spsolve(sN_uz, Nrhs + sN_eps.dot(eps))
            eff_stress_zz = -(labda + 2*mu)*eps
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            time[k] = t
            k += 1
    
    return coeff, time

def new_1d(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = 3.0, dt=0.01):
    ###Infinitely many solutions
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_eps = gK * copy.deepcopy(A)
    alpha_P = gK * p * beta * copy.deepcopy(A)

    #Numerical model + neumann boundary conditions
    Mnew_11 = copy.deepcopy(alpha_eps) + dt*lm*copy.deepcopy(B)
    Mnew_12 = copy.deepcopy(alpha_P) - dt*copy.deepcopy(C)
    Mnew_21 = copy.deepcopy(alpha_eps)
    Mnew_22 = copy.deepcopy(alpha_P) + dt*copy.deepcopy(B)
    
    Mold_11 = copy.deepcopy(alpha_eps)
    Mold_12 = copy.deepcopy(alpha_P)
    Mold_21 = copy.deepcopy(alpha_eps)
    Mold_22 = copy.deepcopy(alpha_P)

    Neps = copy.deepcopy(A)
    Nuz = copy.deepcopy(C)
    Nrhs = copy.deepcopy(zero_vector)
    
    #Dirichlet boundary conditions  
    #BC: P = Fzz at z=0 in 2nd eq
    Mnew_21[-1,:] = 0.0
    Mnew_22[-1,:] = 0.0
    Mold_21[-1,:] = 0.0
    Mold_22[-1,:] = 0.0
    Mnew_22[-1,-1] = 1.0

    #BC: u_z = 0 at z=-n_z in 3rd eq
    Neps[0,:] = 0.0
    Nuz[0,:] = 0.0
    Nuz[0, 0] = 1.0
    Nrhs[0] = 0.0
    
    #Make matrices
    Mnew_1 = np.hstack((Mnew_11, Mnew_12))
    Mnew_2 = np.hstack((Mnew_21, Mnew_22))
    Mnew = np.vstack((Mnew_1, Mnew_2))
    
    Mold_1 = np.hstack((Mold_11, Mold_12))
    Mold_2 = np.hstack((Mold_21, Mold_22))
    Mold = np.vstack((Mold_1, Mold_2))

    Mrhs = np.hstack((zero_vector, zero_vector))
    
    #Make matrices sparse
    sM_new = sp.sparse.csr_matrix(Mnew)
    sM_old = sp.sparse.csr_matrix(Mold)
    sN_uz = sp.sparse.csr_matrix(Nuz)
    sN_eps = sp.sparse.csr_matrix(Neps)

    #Initialise variables
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    num_t = 2 #amount of timesteps that will be saved
    num_elt = np.shape(B)[1] #number of elements
    coeff = np.zeros((4,num_t,num_elt))
    time = np.zeros(num_t)
    
    k=0
    #start time for-loop
    for t in np.arange(dt, time_stop+dt, dt):        
        Mrhs[-1] = fnc(t)
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
    
        if t == (time_stop/10.25)/4 or t == time_stop:
            eps = S[0:num_elt]
            P = S[num_elt:]
            uz = sp.sparse.linalg.spsolve(sN_uz, Nrhs + sN_eps.dot(eps))
            eff_stress_zz = -(labda + 2*mu)*eps
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            time[k] = t
            k += 1
    
    return coeff, time

#--------------------end of functions and start of main--------------------------------------------------------
if __name__ == "__main__":
    #Initialize varibales
    Z, m, q, k, n_q, integrate = parameters_1d.ini_fem()
    #Set soil parameters
    beta, K_s, gamma_w, p, mu, labda = parameters_1d.parameters(num_types=1)
    step_size = Z/m      
    x_lst = np.linspace(-Z,0, m+1)
    print("stepsize = ", step_size)

    ###Initialize directories
    mesh = fem_functions_1d.create_mesh(x_lst)
    param_map = fem_functions_1d.create_param_map(mesh)
    space = fem_functions_1d.create_fe_space(q, k, mesh)
    ref_data = fem_functions_1d.create_ref_data(n_q, q, integrate)

    #Initialize FEM matrices
    A, B, C = fem_functions_1d.assemble_fe_problem(mesh, space, ref_data, param_map)

    #set time step
    dt = 0.01
    print("timestep = ", dt)

    #Choose model
    model = input("\nWhich model do you want? Choose the simplified new model (New_S), 'strict' new model (strict_New), new model (New) or Biot (Biot)? ")
    while model not in ['New_S', 'new_S', 'strict_new', 'strict_New', 'New', 'new', 'Biot', 'biot']:
         model = input("\nNot possible. Choose again: New or Biot: ")
    if model == 'New_S' or model == 'new_S':
        alg_time = newS_1d
    elif model == 'strict_new' or model == 'strict_New':
        alg_time = strict_new_1d
    elif model == 'New' or model == 'new':
        alg_time = new_1d
    elif model == 'Biot' or model == 'biot':
        alg_time = biot_1d
    
    #Choose density or the saturation degree of sand
    data_part = input("\nDo you want to investigate the density or the degree of saturation or extra? Choose density or saturation or extra: ")
    while data_part not in ['Density', 'density', 'Saturation', 'saturation', 'Extra', 'extra']:
         data_part = input("\nNot possible. Choose again: density or saturation or extra: ")

    #Choose part a, b, c, d
    data_plot = input("\nWhich parameters do you want? Choose a, b, c, d or e: ")
    while data_plot not in ['a', 'b', 'c', 'd', 'e']:
         data_plot = input("\nNot possible. Choose again: a, b, c, d or e: ")

    #Data set B. Liu, D.-S. Jeng, G. Ye, and B. Yang, Laboratory study for pore pressures in sandy deposit under wave loading, Ocean Engineering, vol. 106, pp. 207–219, 2015, ISSN: 0029-8018. DOI:https://doi.org/10.1016/j.oceaneng.2015.06.029.

    #Set wave parameters depending on the part
    if data_plot == "a":
        T = 15
        time_stop = 10 * T + T/4
        H = 3.5
    elif data_plot == "b":
        T = 9
        time_stop = 10 * T + T/4
        H = 2.5
    elif data_plot == "c":
        T = 9
        time_stop = 10 * T + T/4
        H = 3.5
    elif data_plot == "d":
        T = 9
        time_stop = 10 * T + T/4
        H = 1.23
    else:
        T = 9
        time_stop = 10 * T + T/4
        H = 3.5
        
    fnc = fem_functions_1d.f_lab(gamma_w, T, H, Nc=10)

    #Solve model for different densities and degrees of saturation
    set_coeff = []
    if data_part == "saturation":
        label1 = 'Low saturation'
        label2 = 'High saturation'
        P_0 = 10**5 #[Pa] atmos. pressure 10**5
        beta_0 = 0.5 * 10**(-9) #[m^2/N]
        s_lst = [0.994,1.0] #[0.951, 0.996]
        for i in range(2):
            s = s_lst[i]
            beta = (1.0-s)/P_0 + s*beta_0 
            coeff, time = alg_time(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = time_stop, dt = dt)
            coeff_sol, dcoeff_sol, z = fem_functions_1d.sum_coeff(mesh, space ,ref_data, param_map, coeff) #determine solution seperately
            set_coeff += [copy.deepcopy(coeff_sol)]
            
    elif data_part == "density":
        label1 = "Dense sand"
        label2 = "Loose sand"
        p_lst = [0.387, 0.425]
        mu_lst = [1.27*10**(7), 1.27*10**(6)]
        vp = 0.3
        c_lst = np.array([0.738,0.467])
        K_s_lst = [1.4*10**(-4), 2.1*10**(-3)]
        
        for i in range(2):
            mu = mu_lst[i]
            E = mu * 2 * (1+vp) #[N/m^2] or [Pa]
            labda = (E*vp) / ((1+vp)*(1-2*vp))
            p = p_lst[i]
            K_s = K_s_lst[i]
            coeff, time = alg_time(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = time_stop, dt = dt)
            coeff_sol, dcoeff_sol, z = fem_functions_1d.sum_coeff(mesh, space ,ref_data, param_map, coeff) #determine solution seperately
            set_coeff += [copy.deepcopy(coeff_sol)]

    else:
        label1 = 'Loose sand and high saturation'
        label2 = ' '
        K_s = 2.1*10**(-4)
        coeff, time = alg_time(A, B, C, beta, K_s, gamma_w, p, mu, labda, fnc, time_stop = time_stop, dt = dt)
        coeff_sol, dcoeff_sol, z = fem_functions_1d.sum_coeff(mesh, space ,ref_data, param_map, coeff) #determine solution seperately
        set_coeff += [copy.deepcopy(coeff_sol)]
        
    set_coeff = np.array(set_coeff)

    #Compressible or incompressible?
    if beta == 0.0 or beta == 0.5 * 10**(-9):
        compr = "Incompressible"
    else:
        compr = "Compressible"
        
    #Determine and plot (normalised) eps, P, u_z
    plot_functions_1d.plot_pressure_norm_data(z[-1], set_coeff[:,2,-1,:,:], time[-1], gamma_w, H, Z, data_part, data_plot, label1, label2)
    plot_functions_1d.plot_norm(coeff_sol, dcoeff_sol, z, Z, time, gamma_w, compr) #set_coeff[2]
    plot_functions_1d.plot(coeff_sol, dcoeff_sol, z, time, compr) #set_coeff[2]


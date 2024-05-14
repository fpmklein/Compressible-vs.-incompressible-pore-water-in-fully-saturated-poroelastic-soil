import numpy as np
import copy
import scipy as sp
import parameters_1d
import fem_functions_1d
import plot_functions_1d
from data1d_paper import data

def biot_1d(A, dA_eps, dA_P, B, dB_eps, C, dC_eps, lm, fnc, time_stop = 3.0, dt=0.01):
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #Numerical model + neumann boundary conditions
    #BC: (labda + 2*mu) eps= 0 at z=0 -> duz/dz = 0 (neumann)
    Mnew_11 = copy.deepcopy(dB_eps)
    Mnew_12 = copy.deepcopy(C)
    Mnew_21 = copy.deepcopy(dC_eps)
    Mnew_22 = copy.deepcopy(dA_P + dt*B)
    
    Mold_11 = copy.deepcopy(zero_matrix)
    Mold_12 = copy.deepcopy(zero_matrix)
    Mold_21 = copy.deepcopy(dC_eps)
    Mold_22 = copy.deepcopy(dA_P)

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

    #BC: uz = 0 at z=-n_z
    Mnew_11[0,:] = 0.0
    Mnew_12[0,:] = 0.0
    Mold_11[0,:] = 0.0
    Mold_12[0,:] = 0.0
    Mnew_11[0,0] = 1.0

    #BC: eps = 0 at z=-n_z in 3rd eq
    Neps[0,:] = 0.0
    Nuz[0,:] = 0.0
    Neps[0, 0] = 1.0
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
    time_lst, P1_lst = fnc[0], fnc[1]
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    
    div = 50
    num_t = len(time_lst)//div + 1
    num_elt = np.shape(B)[1]  
    coeff = np.zeros((5,num_t,num_elt))
    time = np.zeros(num_t)
    time[0] = time_lst[0]
    
    k=1
    #start time for-loop
    for i in range(1,len(time_lst)):
        Mrhs[-1] = P1_lst[i]*gamma_w
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
        if i%div==0:
            uz_old = S_old[0:num_elt]
            uz = S[0:num_elt]
            P_old = S_old[num_elt:]
            P = S[num_elt:]
            eps_old = sp.sparse.linalg.spsolve(sN_eps, Nrhs + sN_uz.dot(uz_old))
            eps = sp.sparse.linalg.spsolve(sN_eps, Nrhs + sN_uz.dot(uz))
            volume_balance = sp.sparse.csr_matrix(dA_eps).dot(eps-eps_old) + sp.sparse.csr_matrix(dA_P).dot(P-P_old) + dt*sp.sparse.csr_matrix(dB_eps).dot(eps)
            eff_stress_zz = eps
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            coeff[4,k,:] = volume_balance
            time[k] = time_lst[i]
            k+=1            
    return coeff, time

def newS_1d(A, dA_eps, dA_P, B, dB_eps, C, dC_eps, lm, fnc, time_stop = 3.0, dt=0.01):
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #Numerical model + neumann boundary conditions
    Mnew = copy.deepcopy(dA_P + dt*B)
    Mold = copy.deepcopy(dA_P)
    Mrhs = copy.deepcopy(zero_vector)
    
    Neps = copy.deepcopy(A) 
    Nuz = copy.deepcopy(C)
    Nrhs = copy.deepcopy(zero_vector)

    #Dirichlet boundary conditions   
    #BC: P = Fzz at z=0 in 1st eq
    Mnew[-1,:] = 0.0
    Mold[-1,:] = 0.0
    Mnew[-1,-1] = 1.0

    #u_z = 0 at z=-n_z in 2nd eq
    Neps[0,:] = 0.0
    Nuz[0,:] = 0.0
    Nuz[0, 0] = 1.0

    #Make matrices sparse
    sM_new = sp.sparse.csr_matrix(Mnew)
    sM_old = sp.sparse.csr_matrix(Mold)
    sN_uz = sp.sparse.csr_matrix(Nuz)
    sN_eps = sp.sparse.csr_matrix(Neps)

    #Initialise variables
    time_lst, P1_lst = fnc[0], fnc[1]
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    
    div = 50
    num_t = len(time_lst)//div + 1
    num_elt = np.shape(B)[1]  
    coeff = np.zeros((5,num_t,num_elt))
    time = np.zeros(num_t)
    time[0] = time_lst[0]
    
    k=1
    num_elt_mid = 101 #boundary between two layers of soil, see assemble_fe_problem_2types in fem_functions_1d
    #start time for-loop
    for i in range(1,len(time_lst)):
        Mrhs[-1] = P1_lst[i]*gamma_w
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
        if i%div==0:
            eps_old = np.hstack(( (1.0/(labda[0] + 2.0*mu[0])) * S_old[0:num_elt_mid], (1.0/(labda[1] + 2.0*mu[1])) * S_old[num_elt_mid:] ))
            eps = np.hstack(( (1.0/(labda[0] + 2.0*mu[0])) * S[0:num_elt_mid], (1.0/(labda[1] + 2.0*mu[1])) * S[num_elt_mid:] ))
            P_old = S_old
            P = S
            uz = sp.sparse.linalg.spsolve(sN_uz, Nrhs + sN_eps.dot(eps))
            eff_stress_zz = S
            volume_balance = sp.sparse.csr_matrix(dA_eps).dot(eps-eps_old) + sp.sparse.csr_matrix(dA_P).dot(P-P_old) + dt*sp.sparse.csr_matrix(dB_eps).dot(eps)
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            coeff[4,k,:] = volume_balance
            time[k] = time_lst[i]
            k+=1            
    return coeff, time

    
def strict_new_1d(A, dA_eps, dA_P, B, dB_eps, C, dC_eps, lm, fnc, time_stop = 3.0, dt=0.01):
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #Numerical model + neumann boundary conditions
    Mnew_11 = copy.deepcopy(dA_eps + dt*dB_eps)
    Mnew_12 = copy.deepcopy(dA_P)
    Mnew_21 = copy.deepcopy(dA_eps)
    Mnew_22 = copy.deepcopy(dA_P + dt*B)
    Mold_11 = copy.deepcopy(dA_eps)
    Mold_12 = copy.deepcopy(dA_P)
    Mold_21 = copy.deepcopy(dA_eps)
    Mold_22 = copy.deepcopy(dA_P)

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

    #BC: (labda + 2*mu) eps= P at z=0
    Mnew_11[-1,:] = 0.0
    Mnew_12[-1,:] = 0.0
    Mnew_11[-1,:] = 0.0
    Mold_12[-1,:] = 0.0
    Mnew_11[-1,-1] = lm[-1]
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
    time_lst, P1_lst = fnc[0], fnc[1]
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    
    div = 50
    num_t = len(time_lst)//div + 1
    num_elt = np.shape(B)[1]  
    coeff = np.zeros((5,num_t,num_elt))
    time = np.zeros(num_t)
    time[0] = time_lst[0]
    
    k=1
    #start time for-loop
    for i in range(1,len(time_lst)):
        Mrhs[-1] = P1_lst[i]*gamma_w
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
        if i%div==0:
            eps_old = S_old[0:num_elt]
            eps = S[0:num_elt]
            P_old = S_old[num_elt:]
            P = S[num_elt:]
            uz = sp.sparse.linalg.spsolve(sN_uz, Nrhs + sN_eps.dot(eps))
            eff_stress_zz = eps
            volume_balance = sp.sparse.csr_matrix(dA_eps).dot(eps-eps_old) + sp.sparse.csr_matrix(dA_P).dot(P-P_old) + dt*sp.sparse.csr_matrix(dB_eps).dot(eps)
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            coeff[4,k,:] = volume_balance
            time[k] = time_lst[i]
            k+=1            

    return coeff, time

def new_1d(A, dA_eps, dA_P, B, dB_eps, C, dC_eps, lm, fnc, time_stop = 3.0, dt=0.01):
    ###Infinitely many solutions
    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #Numerical model + neumann boundary conditions
    Mnew_11 = copy.deepcopy(dA_eps) + dt*copy.deepcopy(dB_eps)
    Mnew_12 = copy.deepcopy(dA_P) - dt*copy.deepcopy(C)
    Mnew_21 = copy.deepcopy(dA_eps)
    Mnew_22 = copy.deepcopy(dA_P) + dt*copy.deepcopy(B)
    Mold_11 = copy.deepcopy(dA_eps)
    Mold_12 = copy.deepcopy(dA_P)
    Mold_21 = copy.deepcopy(dA_eps)
    Mold_22 = copy.deepcopy(dA_P)

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
    time_lst, P1_lst = fnc[0], fnc[1]
    S = np.zeros(np.shape(Mold)[1])
    uz = np.zeros(np.shape(Nuz)[1])
    
    div = 50
    num_t = len(time_lst)//div + 1
    num_elt = np.shape(B)[1]  
    coeff = np.zeros((5,num_t,num_elt))
    time = np.zeros(num_t)
    time[0] = time_lst[0]
    
    k=1
    #start time for-loop
    for i in range(1,len(time_lst)):
        Mrhs[-1] = P1_lst[i]*gamma_w
        S_old = copy.deepcopy(S)
        S = sp.sparse.linalg.spsolve(sM_new, Mrhs + sM_old.dot(S_old))
        if i%div==0:
            eps_old = S_old[0:num_elt]
            eps = S[0:num_elt]
            P_old = S_old[num_elt:]
            P = S[num_elt:]
            uz = sp.sparse.linalg.spsolve(sN_uz, Nrhs + sN_eps.dot(eps))
            eff_stress_zz = eps
            volume_balance = sp.sparse.csr_matrix(dA_eps).dot(eps-eps_old) + sp.sparse.csr_matrix(dA_P).dot(P-P_old) + dt*sp.sparse.csr_matrix(dB_eps).dot(eps)
            coeff[0,k,:] = eff_stress_zz
            coeff[1,k,:] = eps
            coeff[2,k,:] = P
            coeff[3,k,:] = uz
            coeff[4,k,:] = volume_balance
            time[k] = time_lst[i]
            k+=1            

    return coeff, time

#--------------------end of functions and start of main--------------------------------------------------------
    
if __name__ == "__main__":
    #Initialize varibales
    Z, m, q, k, n_q, integrate = parameters_1d.ini_fem()
    beta, K_s, gamma_w, p, mu, labda = parameters_1d.parameters(num_types=2)
    Z = 2.0
    m = 2000
    dz = Z/m      
    x_lst = np.linspace(-Z,0, m+1)#[::-1]

    ###Initialize directories
    mesh = fem_functions_1d.create_mesh(x_lst)
    param_map = fem_functions_1d.create_param_map(mesh)
    space = fem_functions_1d.create_fe_space(q, k, mesh)
    ref_data = fem_functions_1d.create_ref_data(n_q, q, integrate)

    ###print step in space
    print("stepsize = ", dz)
    
    #set time step
    dt = 0.01
    print("timestep = ", dt)        

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_1 = gK
    alpha_2 = gK * p * beta
    d = gK * (p * beta + 1.0/lm)
    alpha = np.array([alpha_1, alpha_2])
    d_alpha = np.array([d,d])
    
    #Choose model
    model = input("\nWhich model do you want? Choose the simplified new model (New_S), 'strict' new model (strict_New), new model (New) or Biot (Biot)? ")
    while model not in ['New_S', 'new_S', 'strict_new', 'strict_New', 'New', 'new', 'Biot', 'biot']:
         model = input("\nNot possible. Choose again: New or Biot: ")

    if model == 'New_S' or model == 'new_S':
        alg_time = newS_1d
        #make recurring matrices
        A, dA_1, dA_2, B, dB_1, C, dC_1 = fem_functions_1d.assemble_fe_problem_2types(mesh, space, ref_data, param_map, lm, alpha)

    elif model == 'strict_new' or model == 'strict_New':
        alg_time = strict_new_1d
        #make recurring matrices
        A, dA_1, dA_2, B, dB_1, C, dC_1 = fem_functions_1d.assemble_fe_problem_2types(mesh, space, ref_data, param_map, lm, alpha)

    elif model == 'New' or model == 'new':
        alg_time = new_1d
        #make recurring matrices
        A, dA_1, dA_2, B, dB_1, C, dC_1 = fem_functions_1d.assemble_fe_problem_2types(mesh, space, ref_data, param_map, lm, alpha)

    elif model == 'Biot' or model == 'biot':
        alg_time = biot_1d
        #make recurring matrices
        A, dA_1, dA_2, B, dB_1, C, dC_1 = fem_functions_1d.assemble_fe_problem_2types(mesh, space, ref_data, param_map, lm, d_alpha)

    #Choose part a, b, c, d
    data_name = input("\nWhich input load do you want? Choose a, b, c, or d: ")
    while data_name not in ['a', 'b', 'c', 'd']:
         data_name = input("\nNot possible. Choose again: a, b, c, or d: ")
    data_name = 'boat_'+ data_name
    
    #boat case
    if data_name == "boat_a":
        date = "(26-10-21 09:13:30)"
    elif data_name == "boat_b":
        date = "(27-10-21 17:49:28)"
    elif data_name == "boat_c":
        date = "(28-10-21 15:41:56)"
    elif data_name == "boat_d":
        date = "(01-11-21 09:01:03)"

    #Determine data
    t_lst, P1_lst, P2_lst, P3_lst, P4_lst, xticks = data(data_name)
    P1_lst = P1_lst - P1_lst[0]
    P2_lst = P2_lst - P2_lst[0]
    P4_lst = P4_lst - P4_lst[0]
    time_stop = t_lst[-1] - t_lst[0] #if t_start = 0, then t_end = t_lst_end - t_lst_begin
    time, P = fem_functions_1d.f_boat(dt, t_lst, P1_lst)

    #Solve model
    coeff, time = alg_time(A, dA_1, dA_2, B, dB_1, C, dC_1, lm, [time,P], time_stop = time_stop, dt = dt)
    coeff_sol, dcoeff_sol, z = fem_functions_1d.sum_coeff_2types(mesh, space, ref_data, param_map, coeff, labda + 2*mu) #determine solution seperately
        
    #Determine and plot eps, P, u_z
    if beta == 0 or beta == 0.5 * 10**(-9):
        compr = "Incompressible"
    else:
        compr = "Compressible"

    dz = Z/m
    num_m = (m-1) - np.array([0, int(0.2/dz), int(0.3/dz)])
    num_nq = (n_q-1)-np.array([int(num_m[0]/n_q), int(num_m[1]/n_q), int(num_m[2]/n_q)])
    c = np.array([coeff_sol[:,:,num_m[0],num_nq[0]], coeff_sol[:,:,num_m[1],num_nq[1]], coeff_sol[:,:,num_m[2],num_nq[2]]])
    dc = np.array([dcoeff_sol[:,:,num_m[0],num_nq[0]], dcoeff_sol[:,:,num_m[1],num_nq[1]], dcoeff_sol[:,:,num_m[2],num_nq[2]]])
    
    plot_functions_1d.plot_pressure_data(time, c[:,2,:], gamma_w, data_name, date)
##    plot_functions_1d.plot_others_time(time, c, gamma_w, labda[-1] + 2.0*mu[-1], Z, data_name, date, model=0)
##    plot_functions_1d.plot_others_time_der(time, dc, gamma_w, labda[-1] + 2.0*mu[-1], Z, data_name, date, model=0)
##    plot_functions_1d.plot_volume_balance(time, c[:,4,:], gamma_w, data_name, date) 
##    plot_functions_1d.plot(coeff_sol[:,-1:,:,:] dcoeff_sol[:,-1:,:,:], z, time, compr = compr)

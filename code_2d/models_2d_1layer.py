#1d_fem code for verification of the 1D Finite Element Method, assignment 1.3, Femke Klein
import numpy as np
import copy
import scipy as sp
import parameters_2d
import fem_functions_2d
import plot_functions_2d

def biot_2d(A, B, C_x, C_z, Mrhs, Mrhs_x, Mrhs_z, rhs, beta, K_s, gamma_w, p, mu, labda, n_x, n_z, time_stop =  3.0, dt = 0.01, bc = ""):
    #Determine the total dimension
    n  = n_x*n_z

    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_ux = copy.deepcopy(gK * C_x)
    alpha_uz = copy.deepcopy(gK * C_z)
    alpha_eps = copy.deepcopy(gK * A)
    alpha_P = copy.deepcopy(gK * p * beta * A)

    #Numerical model + neumann boundary conditions
    M_new_11 = copy.deepcopy(alpha_P) + dt*copy.deepcopy(B)
    M_new_12 = copy.deepcopy(alpha_ux)  
    M_new_13 = copy.deepcopy(alpha_uz)
    
    M_new_21 = copy.deepcopy(C_x) 
    M_new_22 = lm*copy.deepcopy(B)
    if bc == "shear":
        M_new_23 = lm*copy.deepcopy(Mrhs_x)
    elif bc == "vorticity":
        M_new_23 = -lm*copy.deepcopy(Mrhs_x)
    else: #zero
        M_new_23 = copy.deepcopy(zero_matrix)

    M_new_31 = copy.deepcopy(C_z)
    M_new_32 = labda*copy.deepcopy(Mrhs_x)
    M_new_33 = lm*copy.deepcopy(B)
    
    M_old_11 = copy.deepcopy(alpha_P) 
    M_old_12 = copy.deepcopy(alpha_ux) 
    M_old_13 = copy.deepcopy(alpha_uz)
    
    M_old_21 = copy.deepcopy(zero_matrix) 
    M_old_22 = copy.deepcopy(zero_matrix) 
    M_old_23 = copy.deepcopy(zero_matrix)
    
    M_old_31 = copy.deepcopy(zero_matrix)
    M_old_32 = copy.deepcopy(zero_matrix)
    M_old_33 = copy.deepcopy(zero_matrix)

    M_rhs_1 = copy.deepcopy(zero_vector)
    M_rhs_2 = copy.deepcopy(zero_vector)
    M_rhs_3 = copy.deepcopy(zero_vector)

    
    #Dirichlet boundary conditions  
    for i in range(n_x):
        ind = -i-1
        #BC: P = Fzz at z=0 in 1st eq
        M_new_11[ind,:] = 0.0
        M_new_12[ind,:] = 0.0
        M_new_13[ind,:] = 0.0
        M_old_11[ind,:] = 0.0
        M_old_12[ind,:] = 0.0
        M_old_13[ind,:] = 0.0
        M_new_11[ind,ind] = 1.0

        #BC: uz = 0 at z=-n_z in 3rd eq
        M_new_31[i,:] = 0.0
        M_new_32[i,:] = 0.0
        M_new_33[i,:] = 0.0
        M_old_31[i,:] = 0.0
        M_old_32[i,:] = 0.0
        M_old_33[i,:] = 0.0
        M_new_33[i,i] = 1.0

    for i in range(n_z):
        #BC: u_x=0 at x=0 in 2nd eq
        ind_0 = i*n_x
        M_new_21[ind_0,:] = 0.0
        M_new_22[ind_0,:] = 0.0
        M_new_23[ind_0,:] = 0.0
        M_old_21[ind_0,:] = 0.0
        M_old_22[ind_0,:] = 0.0
        M_old_23[ind_0,:] = 0.0
        M_new_22[ind_0,ind_0] = 1.0

        #BC: u_x=0 at x=L in 2nd eq
        ind_L = n_x-1 + i*n_x
        M_new_21[ind_L,:] = 0.0
        M_new_22[ind_L,:] = 0.0
        M_new_23[ind_L,:] = 0.0
        M_old_21[ind_L,:] = 0.0
        M_old_22[ind_L,:] = 0.0
        M_old_23[ind_L,:] = 0.0
        M_new_22[ind_L,ind_L] = 1.0

    #Make matrices and vector    
    M_new_1 = np.hstack((M_new_11, M_new_12, M_new_13))
    M_new_2 = np.hstack((M_new_21, M_new_22, M_new_23))
    M_new_3 = np.hstack((M_new_31, M_new_32, M_new_33))
    M_new = np.vstack((M_new_1, M_new_2, M_new_3))

    M_old_1 = np.hstack((M_old_11, M_old_12, M_old_13))
    M_old_2 = np.hstack((M_old_21, M_old_22, M_old_23))
    M_old_3 = np.hstack((M_old_31, M_old_32, M_old_33))
    M_old = np.vstack((M_old_1, M_old_2, M_old_3))

    M_rhs = np.hstack((M_rhs_1, M_rhs_2, M_rhs_3))

    N_eps = copy.deepcopy(A)
    N_eps_ux = copy.deepcopy(C_x)
    N_eps_uz = copy.deepcopy(C_z)

    N_omega = copy.deepcopy(A)
    N_omega_ux = copy.deepcopy(C_z)
    N_omega_uz = copy.deepcopy(C_x)

    #convert dense to sparse
    sM_new = sp.sparse.csc_matrix(M_new)
    sM_old = sp.sparse.csc_matrix(M_old)
    sN_eps = sp.sparse.csc_matrix(N_eps)
    sN_eps_ux = sp.sparse.csc_matrix(N_eps_ux)
    sN_eps_uz = sp.sparse.csc_matrix(N_eps_uz)
    sN_omega = sp.sparse.csc_matrix(N_omega)
    sN_omega_ux = sp.sparse.csc_matrix(N_omega_ux)
    sN_omega_uz = sp.sparse.csc_matrix(N_omega_uz)
    print("\n(time int) Matrices made sparse\n")
    
    #Initialise variables
    num_t = 2 #amount of time steps saved
    num_elt = len(M_rhs_1) #number of elements
    S = np.zeros(np.shape(M_old)[1])
    coeff = np.zeros((8,num_t,num_elt))
    time = np.zeros(num_t)
    
    num = 0
    x_lst = np.linspace(0,L,n_x)
    #Start time for-loop
    sM_new = sp.sparse.linalg.inv(sM_new)
    time_lst = np.arange(0, time_stop+dt, dt)
    for i in range(1,len(time_lst)):
        t = i*dt
        M_rhs[num_elt-n_x:num_elt] = fem_functions_2d.f_lab(t, x_lst, gamma_w)
        
        S_old = copy.deepcopy(S)
        #S = sp.sparse.linalg.spsolve(sM_new, M_rhs + sM_old.dot(S_old)) #
        S = sM_new.dot(M_rhs + sM_old.dot(S_old))
        if t == time_lst[-2] or t == time_lst[-1]:
            P = S[0:num_elt]
            ux = S[num_elt:2*num_elt]
            uz = S[2*num_elt:3*num_elt]
            
            eps = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), sp.sparse.csc_matrix(C_x).dot(ux) + sp.sparse.csc_matrix(C_z).dot(uz))
            omega = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), sp.sparse.csc_matrix(C_z).dot(ux) - sp.sparse.csc_matrix(C_x).dot(uz))
            sigma_zz = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), -labda*sN_eps.dot(eps)-2.0*mu*sN_eps_uz.dot(uz))
            sigma_xz = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), -mu*(sN_eps_uz.dot(ux)+sN_eps_ux.dot(uz)))
            
            coeff[0,num] = sigma_zz
            coeff[1,num] = eps
            coeff[2,num] = P
            coeff[3,num] = ux
            coeff[4,num] = uz
            coeff[5,num] = omega
            coeff[6,num] = sigma_xz
            
            time[num] = t
            num += 1

    volume_balance =  ((1.0/dt)*sp.sparse.csc_matrix(alpha_P).dot(coeff[2,1]-coeff[2,0]) + (1.0/dt)*sp.sparse.csc_matrix(alpha_eps).dot(coeff[1,1]-coeff[1,0])
                       + lm*sp.sparse.csc_matrix(B).dot(coeff[1,1])
                       + lm*(2*mu/labda)*sp.sparse.csc_matrix(Mrhs_z).dot(coeff[4,1])) #0=-labda eps - 2 mu duz/dz -> eps = -2mu/labda duz/dz
    coeff[7,1] = volume_balance

    print("\nEnd time integration\n")
    
       
    return coeff, time

def newS_2d(A, B, C_x, C_z, Mrhs, Mrhs_x, Mrhs_z, rhs, beta, K_s, gamma_w, p, mu, labda, n_x, n_z, time_stop =  3.0, dt = 0.01, bc = ""):
    #Determine the total dimension
    n  = n_x*n_z

    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    d = gK * (p * beta + 1.0/lm)
    alpha_S = d * copy.deepcopy(A) 

    #Numerical model + neumann boundary conditions
    M_new = copy.deepcopy(alpha_S) + dt*copy.deepcopy(B)
    M_old = copy.deepcopy(alpha_S)
    M_rhs = copy.deepcopy(zero_vector)

    N_eps_x = copy.deepcopy(C_x)
    N_eps_z = copy.deepcopy(C_z) - copy.deepcopy(Mrhs) ##du_z/dz = eps - dux/dx
    N_ux = copy.deepcopy(B)
    N_uz = copy.deepcopy(B) 
    N_uz_x = copy.deepcopy(Mrhs_x)
    N_rhs_x = copy.deepcopy(zero_vector)
    N_rhs_z = copy.deepcopy(zero_vector)
    
    #Dirichlet boundary conditions     
    for i in range(n_x):
        ind = -i-1
        #BC: P = Fzz at z=0 in 1st eq
        M_new[ind,:] = 0.0
        M_old[ind,:] = 0.0
        M_new[ind,ind] = 1.0

        #BC: u_z = 0 at z=-n_z in 3rd eq
        N_eps_z[i,:] = 0.0
        N_uz_x[i,:] = 0.0
        N_uz[i,:] = 0.0
        N_uz[i,i] = 1.0
    
    for i in range(n_z):
        #BC: u_x=0 at x=0 in 2nd eq
        ind_0 = i*n_x
        N_eps_x[ind_0,:] = 0.0
        N_ux[ind_0,:] = 0.0
        N_ux[ind_0,ind_0] = 1.0
        
        #BC: u_x=0 at x=L in 2nd eq
        ind_L = n_x-1 + i*n_x
        N_eps_x[ind_L,:] = 0.0
        N_ux[ind_L,:] = 0.0
        N_ux[ind_L,ind_L] = 1.0

    #convert dense to sparse
    sM_new = sp.sparse.csr_matrix(M_new)
    sM_old = sp.sparse.csr_matrix(M_old)
    
    sN_ux = sp.sparse.csr_matrix(N_ux)
    sN_uz = sp.sparse.csr_matrix(N_uz)

    sN_eps_x = sp.sparse.csr_matrix(N_eps_x)
    sN_eps_z = sp.sparse.csr_matrix(N_eps_z)
    sN_uz_x = sp.sparse.csr_matrix(N_uz_x)
    
    print("\n(time int) Matrices made and boundaries are included.\n")

    #Initialise variables
    num_t = 2 #amount of time steps saved
    num_elt = len(N_rhs_x) #number of elements
    S = np.zeros(np.shape(sM_old)[1])
    coeff = np.zeros((8,num_t,num_elt))
    time = np.zeros(num_t)
    
    num = 0
    x_lst = np.linspace(0,L,n_x)
    #Start time for-loop
    sM_new = sp.sparse.linalg.inv(sM_new)
    time_lst = np.arange(0, time_stop+dt, dt)
    for i in range(1,len(time_lst)):
        t = i*dt
        M_rhs[-n_x:] = fem_functions_2d.f_lab(t, x_lst, gamma_w)
            
        S_old = copy.deepcopy(S)
        #S = sp.sparse.linalg.spsolve(sM_new, M_rhs + sM_old.dot(S_old))
        S = sM_new.dot(M_rhs + sM_old.dot(S_old))
        if t == time_lst[-2] or t == time_lst[-1]:
            print('t = ', t)
            S_new = copy.deepcopy(S)
            eps = S_new/lm
            P = S_new
            
            N_rhs_x = -sN_eps_x.dot(eps) #-rhs*fem_functions_2d.g_lab_partT(t,gamma_w)*(1.0/(2.0*mu))

            ux = sp.sparse.linalg.spsolve(sN_ux, N_rhs_x)
            N_rhs_z = -sN_eps_z.dot(eps) - sN_uz_x.dot(ux)
            uz = sp.sparse.linalg.spsolve(sN_uz, N_rhs_z)

            omega = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), sp.sparse.csc_matrix(C_z).dot(ux) - sp.sparse.csc_matrix(C_x).dot(uz))
            sigma_zz = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), -labda*sp.sparse.csc_matrix(A).dot(eps)-2.0*mu*sp.sparse.csc_matrix(C_z).dot(uz))
            sigma_xz = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), -mu*(sp.sparse.csc_matrix(C_z).dot(ux)+sp.sparse.csc_matrix(C_x).dot(uz)))


            coeff[0,num] = sigma_zz
            coeff[1,num] = eps
            coeff[2,num] = P
            coeff[3,num] = ux
            coeff[4,num] = uz
            coeff[5,num] = omega
            coeff[6,num] = sigma_xz
            
            time[num] = t
            num += 1
            
    volume_balance =  (d*sp.sparse.csc_matrix(A).dot(coeff[2,1]-coeff[2,0])
                       + dt*sp.sparse.csc_matrix(B).dot(coeff[2,1]))
    coeff[7,1] = volume_balance

    print("\nEnd time integration\n")

    return coeff, time

def new_2d(A, B, C_x, C_z, Mrhs, Mrhs_x, Mrhs_z, rhs, beta, K_s, gamma_w, p, mu, labda, n_x, n_z, time_stop = 3.0, dt=0.01, bc = ""):
    #Determine the total dimension
    n  = n_x*n_z

    #Make zero-matrix and zero-vector
    zero_matrix = np.zeros(np.shape(A))
    zero_vector = np.zeros(np.shape(A)[1])

    #make constants
    lm = labda+2.0*mu
    gK = gamma_w / K_s

    #make recurring matrices
    alpha_eps = copy.deepcopy(gK * A) 
    alpha_P = copy.deepcopy(gK * p * beta * A)

    #Numerical model + neumann boundary conditions   
    M_new_11 = (1.0/lm)*copy.deepcopy(alpha_eps) + dt*copy.deepcopy(B)
    M_new_12 = copy.deepcopy(alpha_P) - dt*copy.deepcopy(Mrhs_z)
    M_new_21 = (1.0/lm)*(copy.deepcopy(alpha_eps))
    M_new_22 = (copy.deepcopy(alpha_P) + dt*copy.deepcopy(B))
    
    M_old_11 = (1.0/lm)*copy.deepcopy(alpha_eps)
    M_old_12 = copy.deepcopy(alpha_P)
    M_old_21 = (1.0/lm)*copy.deepcopy(alpha_eps)
    M_old_22 = copy.deepcopy(alpha_P)

    M_rhs_1 = copy.deepcopy(zero_vector)
    M_rhs_2 = copy.deepcopy(zero_vector)

    N_eps_x = copy.deepcopy(C_x)
    N_eps_z = copy.deepcopy(C_z) - copy.deepcopy(Mrhs) #du_z/dz = eps - dux/dx
    N_ux = copy.deepcopy(B)
    N_uz = copy.deepcopy(B)
    N_uz_x = copy.deepcopy(Mrhs_x)
    N_rhs_x = copy.deepcopy(zero_vector)
    N_rhs_z = copy.deepcopy(zero_vector)

    #Dirichlet boundary conditions 
    for i in range(n_x):
        #BC: P = Fzz at z=0 in 2nd eq
        ind = -i-1
        M_new_21[ind,:] = 0.0
        M_new_22[ind,:] = 0.0
        M_old_21[ind,:] = 0.0
        M_old_22[ind,:] = 0.0
        M_new_22[ind,ind] = 1.0

        #BC: u_z = 0 at z=-n_z in 4th eq
        N_eps_z[i,:] = 0.0
        N_uz_x[i,:] = 0.0
        N_uz[i,:] = 0.0
        N_uz[i, i] = 1.0

    for i in range(n_z):
        #BC: u_x=0 at x=0 in 3rd eq
        ind_0 = i*n_x
        N_eps_x[ind_0,:] = 0.0
        N_ux[ind_0,:] = 0.0
        N_ux[ind_0,ind_0] = 1.0

        #BC: u_x=0 at x=L in 3rd eq
        ind_L = n_x-1 + i*n_x
        N_eps_x[ind_L, :] = 0.0
        N_ux[ind_L,:] = 0.0
        N_ux[ind_L,ind_L] = 1.0

    #Make matrices and vector
    M_new_1 = np.hstack((M_new_11, M_new_12))
    M_new_2 = np.hstack((M_new_21, M_new_22))
    M_new = np.vstack((M_new_1, M_new_2))

    M_old_1 = np.hstack((M_old_11, M_old_12))
    M_old_2 = np.hstack((M_old_21, M_old_22))
    M_old = np.vstack((M_old_1, M_old_2))

    M_rhs = np.hstack((M_rhs_1, M_rhs_2))

    #convert dense to sparse
    sM_new = sp.sparse.csc_matrix(M_new)
    sM_old = sp.sparse.csc_matrix(M_old)
    sN_ux = sp.sparse.csr_matrix(N_ux)
    sN_uz = sp.sparse.csr_matrix(N_uz)
    sN_eps_x = sp.sparse.csr_matrix(N_eps_x)
    sN_eps_z = sp.sparse.csr_matrix(N_eps_z)
    sN_uz_x = sp.sparse.csr_matrix(N_uz_x)
    print("\nMatrices made sparse and boundaries are included.\n")
    
    #Initialise variables
    num_t = 2 #amount of time steps saved
    num_elt = len(N_rhs_x) #number of elements
    S = np.zeros(np.shape(sM_old)[1])
    coeff = np.zeros((8,num_t,num_elt))
    time = np.zeros(num_t)
    
    num = 0
    x_lst = np.linspace(0,L,n_x)
    #Start time for-loop
    sM_new = sp.sparse.linalg.inv(sM_new)

    time_lst = np.arange(0, time_stop+dt, dt)
    for i in range(1,len(time_lst)):
        t = i*dt
        M_rhs[-n_x:] = fem_functions_2d.f_lab(t, x_lst, gamma_w) * (1.0/lm)
            
        S_old = copy.deepcopy(S)
        #S = sp.sparse.linalg.spsolve(sM_new, M_rhs + sM_old.dot(S_old)) #np.linalg.solve(M_new, M_rhs + M_old.dot(S_old))
        S = sM_new.dot(M_rhs + sM_old.dot(S_old))
        if t == time_lst[-2] or t == time_lst[-1]:
            eps = S[0:num_elt]
            P = S[num_elt:2*num_elt] * lm
            
            N_rhs_x = -sN_eps_x.dot(eps) #-rhs*fem_functions_2d.g_lab_partT(t,gamma_w)*(1.0/(2.0*mu))           
            ux = sp.sparse.linalg.spsolve(sN_ux, N_rhs_x)
            N_rhs_z = -sN_eps_z.dot(eps) - sN_uz_x.dot(ux)
            uz = sp.sparse.linalg.spsolve(sN_uz, N_rhs_z)

            omega = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), sp.sparse.csc_matrix(C_z).dot(ux) - sp.sparse.csc_matrix(C_x).dot(uz))
            sigma_zz = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), -labda*sp.sparse.csc_matrix(A).dot(eps)-2.0*mu*sp.sparse.csc_matrix(C_z).dot(uz))
            sigma_xz = sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(A), -mu*(sp.sparse.csc_matrix(C_z).dot(ux)+sp.sparse.csc_matrix(C_x).dot(uz)))


            coeff[0,num] = sigma_zz
            coeff[1,num] = eps
            coeff[2,num] = P
            coeff[3,num] = ux
            coeff[4,num] = uz
            coeff[5,num] = omega
            coeff[6,num] = sigma_xz
            
            time[num] = t
            num += 1
            
    volume_balance =  (sp.sparse.csc_matrix(alpha_P).dot(coeff[2,1]-coeff[2,0]) + sp.sparse.csc_matrix(alpha_eps).dot(coeff[1,1]-coeff[1,0])
                       + dt*lm*sp.sparse.csc_matrix(B).dot(coeff[1,1])
                       - dt*sp.sparse.csc_matrix(Mrhs_z).dot(coeff[2,1]))
    coeff[7,1] = volume_balance      

    print("\nEnd time integration\n")
    
    return coeff, time
#--------------------end of functions--------------------------------------------------------
if __name__ == "__main__":
    #Initialize varibales
    L, Z, m_x, m_z, p_x, p_z, k, n_q, integrate = parameters_2d.ini_fem()
    dx = L/m_x 
    dz = Z/m_z
    print(f"\nL, Z, m_x, m_z, deg_x, deg_z, k, n_q = {L, Z, m_x, m_z,p_x, p_z, k, n_q}")
    print("dx = ", dx)
    print("dz = ", dz)
    
    #Set soil parameters
    beta, K_s, gamma_w, p, mu, labda = parameters_2d.parameters(num_types=1)
    print(f"\nK_s, gamma_w, p, mu, labda = {K_s, gamma_w, p, mu, labda}")
    n_x = m_x*(p_x+1) - (m_x - 1)*(k+1)
    n_z = m_z*(p_z+1) - (m_z - 1)*(k+1)

    ###Initialize directories
    mesh = fem_functions_2d.create_uniform_mesh(L, Z, m_x, m_z, p_x, p_z, k)
    ref_data = fem_functions_2d.create_ref_data(n_q, [p_x,p_z], integrate)
    nodes = mesh['nodes']
    subd = mesh['node_numbers']
    map_coeffs = nodes[subd,:]
    geom_map = fem_functions_2d.create_geometric_map(map_coeffs, ref_data)
    space = fem_functions_2d.create_fe_space(mesh, ref_data, geom_map, k)

    #Initialize FEM matrices
    A, B, C_x, C_z, Mrhs, Mrhs_x, Mrhs_z, Nrhs_x, rhs = fem_functions_2d.assemble_fe_problem(mesh, space, ref_data, geom_map)

    #set time step and end time
    time_stop = 2.25#92.25
    dt = 0.01
    print('t_stop, dt = ', time_stop, dt)
    
    #Choose model
    model = input("\nWhich model do you want? Choose the simplified new model (New_S), new model (New) or Biot with shear as bc (Biot_shear), vorticity (Biot_vorticity) or zero (Biot_zero)? ")
    while model not in ['New_S', 'new_S', 'New', 'new', 'Biot_shear', 'biot_shear', 'Biot_vorticity', 'biot_vorticity', 'Biot_zero', 'biot_zero']:
         model = input("\nNot possible. Choose again: New or Biot: ")

    if model == 'New_S' or model == 'new_S':
        alg_time = newS_2d
        model_name = 'New_S'
        bc = ''
    elif model == 'New' or model == 'new':
        alg_time = new_2d
        model_name = 'New'
        bc = ''
    elif model == 'Biot_shear' or model == 'biot_shear':
        alg_time = biot_2d
        model_name = 'Biot_shear'
        bc = 'shear'
    elif model == 'Biot_vorticity' or model == 'biot_vorticity':
        alg_time = biot_2d
        model_name = 'Biot_vorticity'
        bc = 'vorticity'
    elif model == 'Biot_zero' or model == 'biot_zero':
        alg_time = biot_2d
        model_name = 'Biot_zero'
        bc = 'zero'

    #Compressible or incompressible?
    if beta == 0.0 or beta == 0.5 * 10**(-9):
        compr = "Incompressible"
    else:
        compr = "Compressible"
        
    #Determine and plot eps, P, u_x, u_z and volume balance, omega, effective stress sigma'_zz and normal stress sigma_xz
    coeff, time = alg_time(A, B, C_x, C_z, Mrhs, Mrhs_x, Mrhs_z, rhs, beta, K_s, gamma_w, p, mu, labda, n_x, n_z, time_stop=time_stop, dt=dt, bc=bc)
    coeff_sol, dcoeff_sol, x_lst = fem_functions_2d.sum_coeff(mesh, space, ref_data, geom_map, coeff, labda+2.0*mu, mu)
    print("Solutions found")

    plot_functions_2d.plot_2dsolution_heatmap(coeff_sol[:,-1,:], dcoeff_sol[:,:,-1,:], x_lst, model_name)
    plot_functions_2d.plot_2dsolution_heatmap_rest(coeff_sol[:,-1,:], dcoeff_sol[:,:,-1,:], x_lst, model_name)

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl
import parameters_2d
import copy

#Create reference data
def create_ref_data(n_q1, deg, integrate=False):
    p_1 = deg[0]
    p_2 = deg[1]
    
    # reference unit domain
    reference_element = np.array([ [0, 1], [0, 1] ])
    if integrate is False:
        # point for plotting are equispaced on reference element
        
        x = []
        for j in range(n_q1):
            for i in range(n_q1):
                x.append( [ i / (n_q1-1), j / (n_q1-1) ] )
                
        evaluation_points = x
        quadrature_weights = np.zeros((0,))
    else:
        # points (and weights) for integration are computed according to Gauss quadrature
        points, w = gaussquad(n_q1)
        x = 0.5*(points + 1)
        evaluation_points = []
        quadrature_weights = []
        boundary_points = []
        boundary_weights = []
        
        for j in range(len(x)):
            boundary_points.append([x[j], 1])
            boundary_weights.append(w[j]/2)
            for i in range(len(x)):
                evaluation_points.append( [ x[i], x[j] ] )
                quadrature_weights.append( w[i] * w[j] / 4 )
    
    # knots for defining B-splines
    knt_1 = np.concatenate((np.zeros((p_1+1,),dtype=float),np.ones((p_1+1,),dtype=float)),axis=0)
    knt_2 = np.concatenate((np.zeros((p_2+1,),dtype=float),np.ones((p_2+1,),dtype=float)),axis=0)
    
    # reference basis function values
    tmp_1 = [bspl.evaluate_all_bspl(knt_1, p_1, evaluation_points[i][0], p_1, nu=0)
             for i in range(len(evaluation_points))]
    
    tmp_2 = [bspl.evaluate_all_bspl(knt_2, p_2, evaluation_points[i][1], p_2, nu=0)
             for i in range(len(evaluation_points))]

    bd_tmp_1 = [bspl.evaluate_all_bspl(knt_1, p_1, boundary_points[i][0], p_1, nu=0)
                for i in range(len(boundary_points))]

    bd_tmp_2 = [bspl.evaluate_all_bspl(knt_2, p_2, boundary_points[i][1], p_2, nu=0)
             for i in range(len(boundary_points))]
             
    tmp_1 = np.vstack(tmp_1).T
    tmp_2 = np.vstack(tmp_2).T

    bd_tmp_1 = np.vstack(bd_tmp_1).T
    bd_tmp_2 = np.vstack(bd_tmp_2).T

    reference_basis = np.zeros( ( (p_1 + 1)*(p_2 + 1), n_q1 * n_q1  ) )
    boundary_reference_basis = np.zeros( ( (p_1 + 1)*(p_2 + 1), n_q1) )
    for j_2 in range(p_2 + 1):
        for j_1 in range(p_1 + 1):
            i = j_1 + (j_2)*(p_1 + 1)
            arr = []
            bd_arr = []
            for k in range(n_q1 * n_q1):
                arr.append( tmp_1[j_1][k] * tmp_2[j_2][k] )
            for k in range(n_q1):
                bd_arr.append( bd_tmp_1[j_1][k] * bd_tmp_2[j_2][k] )
            reference_basis[i] = arr.copy()
            boundary_reference_basis[i] = bd_arr.copy()
    
    # reference basis 1st derivative function values
    tmp_1_der = [bspl.evaluate_all_bspl(knt_1, p_1, evaluation_points[i][0], p_1, nu=1)
             for i in range(len(evaluation_points))]
    
    tmp_2_der = [bspl.evaluate_all_bspl(knt_2, p_2, evaluation_points[i][1], p_2, nu=1)
             for i in range(len(evaluation_points))]

    bd_tmp_1_der = [bspl.evaluate_all_bspl(knt_1, p_1, boundary_points[i][0], p_1, nu=1)
             for i in range(len(boundary_points))]
    
    bd_tmp_2_der = [bspl.evaluate_all_bspl(knt_2, p_2, boundary_points[i][1], p_2, nu=1)
             for i in range(len(boundary_points))]
             
    tmp_1_der = np.vstack(tmp_1_der).T
    tmp_2_der = np.vstack(tmp_2_der).T

    bd_tmp_1_der = np.vstack(bd_tmp_1_der).T
    bd_tmp_2_der = np.vstack(bd_tmp_2_der).T

    reference_basis_derivatives = np.zeros( ( (p_1 + 1)*(p_2 + 1), n_q1 * n_q1, 2  ) )
    boundary_reference_basis_derivatives = np.zeros( ( (p_1 + 1)*(p_2 + 1), n_q1, 2) )
    for j_2 in range(p_2 + 1):
        for j_1 in range(p_1 + 1):
            i = j_1 + (j_2)*(p_1 + 1)
            arr = []
            bd_arr = []
            for k in range(n_q1 * n_q1):
                arr.append( [ tmp_1_der[j_1][k] * tmp_2[j_2][k], tmp_1[j_1][k] * tmp_2_der[j_2][k] ] )

            for k in range(n_q1):
                bd_arr.append( [ bd_tmp_1_der[j_1][k] * bd_tmp_2[j_2][k], bd_tmp_1[j_1][k] * bd_tmp_2_der[j_2][k] ] )
            reference_basis_derivatives[i] = arr.copy()
            boundary_reference_basis_derivatives[i] = bd_arr.copy()
            
    # store all data and return
    reference_data = {'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'bd_points': boundary_points,
                      'quadrature_weights': quadrature_weights,
                      'bd_weights': boundary_weights,
                      'deg': deg,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives,
                      'bd_reference_basis': boundary_reference_basis,
                      'bd_reference_basis_derivatives': boundary_reference_basis_derivatives
                     }
    
    return reference_data

#create finite element space            
def create_fe_space(mesh, ref_data, geom_map, k=0):
    m_lst = mesh['m']
    m_x = m_lst[0]
    m_z = m_lst[1]
    p = ref_data['deg']
    p_x = p[0]
    p_z = p[1]    
    subd = mesh['node_numbers'] #num_subd x num_nodes_per_subd
    
    #Extraction coefficients
    #unique_points_per_subd = nodes[np.unique(subd)] #num_nodesXdim (num_subd, num_nodes_per_subd)
    num_subd = m_x*m_z
    num_nodes_per_subd = (p_x + 1)*(p_z + 1)
    
    E = np.zeros((num_subd, num_nodes_per_subd, num_nodes_per_subd)) 
    I = np.zeros((num_subd, num_nodes_per_subd), int)

    if k == 0:
        for l in range(num_subd):            
            for j in range(num_nodes_per_subd):
                #index = ind[j]
                E[l,j,j] = 1.0#lst[j]
                I[l,j] = subd[l,j]
    else:
        print("Only k = 0 is implementend.")


    C = np.array(E) #num_subd X I_i X (p_1+1)(p_2+1)
        
    dim_x = m_x*(p_x+1) - (m_x - 1)*(k+1)
    dim_z = m_z*(p_z+1) - (m_z - 1)*(k+1)

    # save and return
    space = {'n': [dim_x,dim_z],
             'supported_bases': I,
             'extraction_coefficients': C
    }
    return space

#Create a uniform mesh
def create_uniform_mesh(L, Z, m_x, m_z, deg_x, deg_z, k=0):
    dim = 2
    corners = np.array([[0,-Z],[L,-Z],[L,0],[0,0]])
    n_x = m_x*(deg_x+1) - (m_x - 1)*(k+1)
    n_z = m_z*(deg_x+1) - (m_z - 1)*(k+1)
    num_nodes = n_x*n_z
    
    #nodes
    nodes = np.zeros((num_nodes, dim))
    step_x = abs(corners[1,0] - corners[3,0]) / (n_x-1)
    step_z = abs(corners[1,1] - corners[3,1]) / (n_z-1)

    for j2 in range(n_z):
        for j1 in range(n_x):
            j = j1 + j2*n_x
            nodes[j,0] = corners[0,0] + j1*step_x #x values
            nodes[j,1] = corners[0,1] + j2*step_z #z values


    # connectivity information (i.e., which bases are non-zero on which element)
    #subdomains
    num_subd = m_x*m_z
    num_nodes_per_subd = (deg_x+1)*(deg_z+1)
    subd = np.zeros((num_subd, num_nodes_per_subd), int)

    if num_nodes_per_subd == 4:
        for j2 in range(m_z):
            for j1 in range(m_x):
                j = j1 + j2*m_x
                node_in_subd = j1 + j2*n_x
                subd[j, 0] = node_in_subd
                subd[j, 1] = node_in_subd + 1
                subd[j, 2] = node_in_subd + n_x
                subd[j, 3] = node_in_subd + n_x + 1
    else:
        print("Only num_nodes_per_subd = 4 is implementend, i.e. p1 = 1, p2 = 1, k = 0.")

    mesh = {'nodes': nodes,
            'node_numbers': subd,
            'm': np.array([m_x,m_z])}
    return mesh

#create geometric map
def create_geometric_map(map_coeffs, ref_data): #make phi, grad phi (jac)
    #map_coeffs.shape = (num_subd, num_nodes_per_subd, dim)
    num_subd = np.shape(map_coeffs)[0]
    num_nodes_per_subd = np.shape(map_coeffs)[1]
    dim = np.shape(map_coeffs)[2]
    
    ref_basis = ref_data['reference_basis']
    ref_basis_der = ref_data['reference_basis_derivatives']
    n_q = np.shape(ref_basis)[1]
    
    
    _map = np.zeros((num_subd, dim, n_q))
    map_derivatives = np.zeros((num_subd,4,n_q))
    imap_derivatives = np.zeros((num_subd,4,n_q))
    lst_det = np.zeros((num_subd,n_q))
    
    for l in range(num_subd):
        x = np.zeros(n_q)
        z = np.zeros(n_q)
        x_xi = np.zeros(n_q)
        z_xi = np.zeros(n_q)
        x_eta = np.zeros(n_q)
        z_eta = np.zeros(n_q)
        for j in range(num_nodes_per_subd):
            x += map_coeffs[l,j,0] * ref_basis[j]
            z += map_coeffs[l,j,1] * ref_basis[j]
            x_xi += map_coeffs[l,j,0] * ref_basis_der[j,:,0]
            z_xi += map_coeffs[l,j,1] * ref_basis_der[j,:,0]
            x_eta += map_coeffs[l,j,0] * ref_basis_der[j,:,1]
            z_eta += map_coeffs[l,j,1] * ref_basis_der[j,:,1]

        _map[l,0] = x
        _map[l,1] = z

        map_derivatives[l,0] = x_xi
        map_derivatives[l,1] = z_xi   
        map_derivatives[l,2] = x_eta
        map_derivatives[l,3] = z_eta   

        det = x_xi*z_eta - z_xi*x_eta
        if det[0] <= 0:
            print("det = ",det)
        imap_derivatives[l,0] = z_eta / det
        imap_derivatives[l,1] = -z_xi / det  
        imap_derivatives[l,2] = -x_eta / det
        imap_derivatives[l,3] = x_xi / det
        lst_det[l] = det
    geom_map = {
                'map': _map,
                'map_derivatives': map_derivatives,
                'imap_derivatives': imap_derivatives,
                'det': lst_det
               }                 
    return geom_map


def problem_A(z,Nj,dNj_1,dNj_2,Nk,dNk_1, dNk_2):
    return Nj*Nk

def problem_B(z,Nj,dNj_1,dNj_2,Nk,dNk_1, dNk_2):
    return dNj_1*dNk_1 + dNj_2*dNk_2

def problem_C_x(z,Nj,dNj_1,dNj_2,Nk,dNk_1, dNk_2):
    return Nj*dNk_1

def problem_C_z(z,Nj,dNj_1,dNj_2,Nk,dNk_1, dNk_2):
    return Nj*dNk_2

def problem_Mrhs(z,Nj,dNj_1,dNj_2,Nk,dNk_1,dNk_2):
    return Nj*Nk

def problem_Mrhs_x(z,Nj,dNj_1,dNj_2,Nk,dNk_1,dNk_2):
    return Nj*dNk_1

def problem_Mrhs_z(z,Nj,dNj_1,dNj_2,Nk,dNk_1,dNk_2):
    return Nj*dNk_2

def problem_rhs(z,Nj,dNj_1,dNj_2):
    return Nj

def problem_rhs_x(z,Nj,dNj_1,dNj_2):
    return dNj_1

#BC: P = F_zz, at surface z=0
def f(t, x, Fc, beta):
    if t < np.pi:
        return (1.0-beta)*Fc*(1.0-np.cos(t)*np.cos(x))
    else:
        return (1.0-beta)*Fc*2.0

#BC: P = F_zz, at surface z=0   
def f_lab(t, x, gamma_w=10**4, L=1.0, T=9, H=3.5, Nc = 10, D=0): #Fzz normal stress
    #T = 9 #[s] wave period
    #H = 3.5 #[m] wave height
    #Nc = 10 #number of waves
    #D = 5.2 #[m] water depth
    #L = 1.0 #D/Nc #[m] distance between each wave
    pressure_z0 = gamma_w*0.5*H*np.cos(2.0*math.pi*(x/L))*np.sin(2.0*math.pi*(t/T)) + gamma_w*D
    
    return pressure_z0

#BC: sigma_xz = F_xz, at surface z=0
def g_lab_partX(x): #Fxz shear stress
    T = 9 #[s] wave period
    H = 3.5 #[m] wave height
    Nc = 1 #number of waves
    D = 5.2 #[m] water depth
    L = 1.0#D/Nc #[m] distance between each wave
    pressure_z0 = math.cos(2*math.pi*(x/L))
    return pressure_z0

#BC: sigma_xz = F_xz, at surface z=0
def g_lab_partT(t, gamma_w): #Fxz shear stress
    T = 9 #[s] wave period
    H = 3.5 #[m] wave height
    Nc = 1 #number of waves
    D = 5.2 #[m] water depth
    L = 1.0#D/Nc #[m] distance between each wave
    pressure_z0 = gamma_w*0.5*H*math.sin(2*math.pi*(t/T))
    return pressure_z0

#debugging: is the sum of the finite basis functions indeed 1?
def debug_sum_fe_basis_fnc(s_lst, tol=10**(-9)): #N_s a numpy array
    print("Debugging: sum of fe basis functions should be 1")
    #s_lst = np.sum(N_s, axis = 0)
    error = 0
    for s in s_lst:
        if abs(s - 1) > tol:
            error += 1
    print("Is sum of finite element basis functions equal to 1?", error==0) #sum over all basis functions must equal 1
    print()

#Assemble finite element basis functions    
def assemble_N_dN(B, dB, E, dxi):
    #E = I_i x (p1+1)(p2+1)
    #B = (p1+1)(p2+1) x nq
    #dB = (p1+1)(p2+1) x nq x 2
    #N_i |omega_l = sum_s E_isl B_sl
    N = np.zeros((np.shape(E)[0], np.shape(B)[1])) #I_i x nq
    dN_1 = np.zeros((np.shape(E)[0], np.shape(dB)[1])) #I_i x nq
    dN_2 = np.zeros((np.shape(E)[0], np.shape(dB)[1])) #I_i x nq
    
    for j0 in range(np.shape(E)[0]): #I_i
        E_jl = E[j0]
        for j1 in range(np.shape(E)[1]): #(p1+1)(p2+1)
            e_jkl = E_jl[j1]
            N[j0] += e_jkl * B[j1,:] 
            dN_1[j0] += e_jkl * (dxi[0]*dB[j1,:,0] + dxi[1]*dB[j1,:,1])
            dN_2[j0] += e_jkl * (dxi[2]*dB[j1,:,0] + dxi[3]*dB[j1,:,1])
    
    #debug_sum_fe_basis_fnc(np.sum(N, axis = 0))
    #debug_sum_fe_basis_fnc(np.sum(E, axis = 0))
    return N, dN_1, dN_2

#Assemble matrices for FEM with 1 layer of soil     
def assemble_fe_problem(mesh, fe_space, ref_data, geom_map):
    # retrieve data
    n_lst = fe_space["n"]
    n_x = n_lst[0]
    n_z = n_lst[1]
    n = n_x*n_z
    
    B = ref_data['reference_basis'] #(p1 + 1)(p2 + 1) × nq
    dB = ref_data['reference_basis_derivatives'] #(p1 + 1)(p2 + 1) × nq x 2 #[dN_tilde/dxi^1, dN_tilde/dxi^2]
    bd_B = ref_data['bd_reference_basis'] #(p1 + 1)(p2 + 1) × nq
    bd_dB = ref_data['bd_reference_basis_derivatives'] #(p1 + 1)(p2 + 1) × nq x 2 #[dN_tilde/dxi^1, dN_tilde/dxi^2]

    w = ref_data['quadrature_weights'] #nq x 1
    lst_det_x = geom_map['det']

    m_lst = mesh['m']
    m_x = m_lst[0]
    m_z = m_lst[1]
    nel = m_x*m_z

    p_lst = ref_data['deg']
    p_1 = p_lst[0]
    p_2 = p_lst[1]
    
    # element-wise assembly loop
    A = np.zeros((n,n))
    C = np.zeros((n,n))
    D_x = np.zeros((n,n))
    D_z = np.zeros((n,n))

    Mrhs = np.zeros((n,n))
    Mrhs_x = np.zeros((n,n))
    Mrhs_z = np.zeros((n,n))
    Nrhs_x = np.zeros((n,n))

    rhs = np.zeros((1,n))
    
    bd_w = int(math.sqrt(len(w)))
    w_z0 = ref_data['bd_weights']
                    
    for ind in range(m_x):
        ###boundary matrices
        i = m_x*(m_z-1) + ind
        # geometry informationdr
        x = geom_map['map'][i] #2 x nq
        dx = geom_map['map_derivatives'][i] #4 x nq 
        dxi = geom_map['imap_derivatives'][i] #4 x nq
        # extraction coefficients and FE basis
        I = fe_space["supported_bases"][i] # 1x#I_i #j-th entry is k
        E = fe_space["extraction_coefficients"][i] # #I_i x (p_1+1)(p_2+1) #(j,:)-th row contains all coefficients e_kli
        
        det_x = lst_det_x[i] #determinant of dx
        phi_x = dx[0]

        N, dN_1, dN_2 = assemble_N_dN(B, dB, E, dxi)
        for j1 in range(len(I)):
            #determine bd_3, which is the boundary at the surface z=0
            tmp_rhs = problem_rhs(x,N[j1],dN_1[j1],dN_2[j1])
            sum_rhs = 0
            for nq in range(bd_w): #nq = nq1*nq1 -> latest nq1 = sqrt(nq) are points of bd at z=0
                 sum_rhs += phi_x[-bd_w+nq] * w_z0[-bd_w+nq] * tmp_rhs[-bd_w+nq] * g_lab_partX(x[0,-bd_w+nq])
            rhs[0,I[j1]] += sum_rhs

            for j2 in range(len(I)):#p_1*(p_2+1)
                tmp_Mrhs = problem_Mrhs(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                tmp_Mrhs_x = problem_Mrhs_x(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                tmp_Mrhs_z = problem_Mrhs_z(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                sum_Mrhs = 0
                sum_Mrhs_x = 0
                sum_Mrhs_z = 0
                for nq in range(bd_w): #nq = nq1*nq1 -> latest nq1 = sqrt(nq) are points of bd at z=0
                    sum_Mrhs += phi_x[-bd_w+nq] * w_z0[-bd_w+nq] * tmp_Mrhs[-bd_w+nq]
                    sum_Mrhs_x += phi_x[-bd_w+nq] * w_z0[-bd_w+nq] * tmp_Mrhs_x[-bd_w+nq]
                    sum_Mrhs_z += phi_x[-bd_w+nq] * w_z0[-bd_w+nq] * tmp_Mrhs_z[-bd_w+nq]
                    
                Mrhs[I[j1], I[j2]] += sum_Mrhs
                Mrhs_x[I[j1], I[j2]] += sum_Mrhs_x
                Mrhs_z[I[j1], I[j2]] += sum_Mrhs_z

        # geometry informationdr
        x = geom_map['map'][ind] #2 x nq
        dx = geom_map['map_derivatives'][ind] #4 x nq 
        dxi = geom_map['imap_derivatives'][ind] #4 x nq
        # extraction coefficients and FE basis
        I = fe_space["supported_bases"][ind] # 1x#I_i #j-th entry is k
        E = fe_space["extraction_coefficients"][ind] # #I_i x (p_1+1)(p_2+1) #(j,:)-th row contains all coefficients e_kli
        
        det_x = lst_det_x[ind] #determinant of dx
        phi_x = dx[0]
        N, dN_1, dN_2 = assemble_N_dN(B, dB, E, dxi) #assemble_N_dN(bd_B, bd_dB, E, dxi[:,-bd_w:])
        for j1 in range(len(I)):
            #determine bd_1, which is the boundary at the bottom z=-Z. -> eta = [0,-1]    
            for j2 in range(len(I)):#p_1*(p_2+1)
                tmp_Nrhs_x = -problem_Mrhs_x(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                sum_Nrhs_x = 0
                for nq in range(bd_w): #nq = nq1*nq1 -> latest nq1 = sqrt(nq) are points of bd at z=0
                    sum_Nrhs_x += phi_x[nq] * w_z0[nq] * tmp_Nrhs_x[nq]

                Nrhs_x[I[j1], I[j2]] += sum_Nrhs_x
                
    ###Intern matrices    
    for i in range(nel): #nel = m
        # geometry information
        x = geom_map['map'][i] #2 x nq
        dx = geom_map['map_derivatives'][i] #4 x nq 
        dxi = geom_map['imap_derivatives'][i] #4 x nq 
        # extraction coefficients and FE basis
        I = fe_space["supported_bases"][i] # 1x#I_i #j-th entry is k
        E = fe_space["extraction_coefficients"][i] # #I_i x (p_1+1)(p_2+1) #(j,:)-th row contains all coefficients e_kli
        det_x = lst_det_x[i]
        
        N, dN_1, dN_2 = assemble_N_dN(B, dB, E, dxi)            
            
        # compute local integrals and add to global matrix
        for j1 in range(len(I)):
            for j2 in range(len(I)):
                tmp_A = problem_A(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                tmp_C = problem_B(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                tmp_D_x = problem_C_x(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                tmp_D_z = problem_C_z(x,N[j1],dN_1[j1],dN_2[j1],N[j2],dN_1[j2],dN_2[j2])
                
                sum_A = 0
                sum_C = 0
                sum_D_x = 0
                sum_D_z = 0
                for nq in range(len(w)):
                    sum_A += det_x[nq] * w[nq] * tmp_A[nq]
                    sum_C += det_x[nq] * w[nq] * tmp_C[nq]
                    sum_D_x += det_x[nq] * w[nq] * tmp_D_x[nq]
                    sum_D_z += det_x[nq] * w[nq] * tmp_D_z[nq]

                            
                A[I[j1],I[j2]] += sum_A
                C[I[j1],I[j2]] += sum_C
                D_x[I[j1],I[j2]] += sum_D_x
                D_z[I[j1],I[j2]] += sum_D_z

    return A, C, D_x, D_z, Mrhs, Mrhs_x, Mrhs_z, Nrhs_x, rhs

#c = sum c_j N_j, 1 layer of soil
def sum_coeff(mesh, fe_space,ref_data, geom_map, coeffs, lm, mu):
    #retrieve data
    m_lst = mesh['m']
    m_x = m_lst[0]
    m_z = m_lst[1]
    nel = m_x*m_z
    
    w = ref_data['quadrature_weights']
    B = ref_data['reference_basis'] #(p1 + 1)(p2 + 1) × nq
    dB = ref_data['reference_basis_derivatives'] #(p1 + 1)(p2 + 1) × nq x 2 #[dN_tilde/dxi^1, dN_tilde/dxi^2]
    
    nq = np.shape(B)[1]

    num_var = np.shape(coeffs)[0]
    num_t = np.shape(coeffs)[1]
    
    lst = np.zeros((num_var,num_t,nel,nq))
    der_lst = np.zeros((num_var,2,num_t,nel,nq))
    x_lst = np.zeros((2, nel, nq))

    for i in range(nel):
        x = geom_map['map'][i] #2xnq
        dx = geom_map['map_derivatives'][i] #4 x nq
        dxi = geom_map['imap_derivatives'][i] #4 x nq
        
        I = fe_space["supported_bases"][i] # 1x#I_i #j-th entry is k
        E = fe_space["extraction_coefficients"][i] # #I_i x (p_1+1)(p_2+1) #(j,:)-th row contains all coefficients e_kli
        
        N, dN_1, dN_2 = assemble_N_dN(B, dB, E, dxi) #N= #I_i x nq

        #0 eps, 1 P, 2 ux, 3 uz for coeffs
        #0 sigma'_zz, 1 eps, 2 P, 3 ux, 4 uz, (5 omega) for coeffs
        for var in range(num_var):
            for t in range(num_t):
                for j in range(len(I)):
                    index = I[j]
                    lst[var,t,i] += coeffs[var,t,index]*N[j]
                    der_lst[var,0,t,i] += coeffs[var,t,index]*dN_1[j,:]
                    der_lst[var,1,t,i] += coeffs[var,t,index]*dN_2[j,:]
            
        x_lst[0,i] = x[0,:]
        x_lst[1,i] = x[1,:]
    return lst, der_lst, x_lst

if __name__ == "__main__":
    ###make the grid
    L = 1
    Z = 2
    m_x = 4
    m_z = 4
    deg_x = 1
    deg_z = 1
    n_q = 4
    k = 0
    integrate = True
    
    #creat dictionaries
    mesh = create_uniform_mesh(L, Z, m_x, m_z, deg_x, deg_z)
    ref_data = create_ref_data(n_q, [deg_x,deg_z], integrate)   
    nodes = mesh['nodes']
    subd = mesh['node_numbers']
    map_coeffs = nodes[subd]
    geom_map = create_geometric_map(map_coeffs, ref_data)
    space = create_fe_space(mesh, ref_data, geom_map, k)

    #Assemble matrices
    #A, B, C_x, C_z = assemble_fe_problem(space, ref_data, geom_map)
    
    num_nodes = np.shape(nodes)[0]
    num_subd = np.shape(subd)[0]
    num_p = np.shape(subd)[1]#(deg_x+1)*(deg_z+1)
    n_x = m_x*(deg_x+1) - (m_x - 1)*(k+1)
    n_z = m_z*(deg_z+1) - (m_z - 1)*(k+1)
    n = n_x*n_z
    num = 0
    for i in range(num_nodes):
        #plt.annotate(num, xy = (nodes[i,0],nodes[i,1]), c = 'blue') #placing the number of the node
        num += 1

    num = 0
    for i in range(num_subd):
        s0 = 0
        s1 = 0
        for j in range(np.shape(subd)[1]):
            s0 += nodes[subd[i,j],0]
            s1 += nodes[subd[i,j],1]
        plt.annotate(num, xy = (s0/num_p,    #placing the number of the subdomain
                                s1/num_p),
                     c = 'black')
        num += 1

    #making red squares
    for i in range(num_subd):
        x = []
        z = []
        for j in range(0,deg_x+1,1):
            x += [nodes[subd[i,j],0]]
            z += [nodes[subd[i,j],1]]
        plt.plot(x, z, c='red', linewidth = 3)
        x = []
        z = []
        for j0 in range(0,deg_x+1,1):
            j = deg_z*(deg_x+1) + j0
            x += [nodes[subd[i,j],0]]
            z += [nodes[subd[i,j],1]]
        plt.plot(x, z, c='red', linewidth = 3)
        x = []
        z = []
        for j in range(0,num_p,deg_x+1):
            x += [nodes[subd[i,j],0]]
            z += [nodes[subd[i,j],1]]
        plt.plot(x, z, c='red', linewidth = 3)
        x = []
        z = []
        for j in range(deg_x,num_p,deg_x+1):
            x += [nodes[subd[i,j],0]]
            z += [nodes[subd[i,j],1]]
        plt.plot(x, z, c='red', linewidth = 3)
             
    plt.show()

    

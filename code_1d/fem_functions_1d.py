import numpy as np
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl
import math

#Create reference data
def create_ref_data(neval, deg, integrate=False):
    # reference unit domain
    reference_element = np.array([0, 1])
    if integrate is False:
        # point for plotting are equispaced on reference element
        x = np.linspace(reference_element[0], reference_element[1], neval)
        evaluation_points = x
        quadrature_weights = np.zeros((0,))
    else:
        # points (and weights) for integration are computed according to Gauss quadrature
        x, w = gaussquad(neval)
        evaluation_points = 0.5*(x + 1)
        quadrature_weights = w/2

    # knots for defining B-splines
    knt = np.concatenate((np.zeros((deg+1,),dtype=float),np.ones((deg+1,),dtype=float)),axis=0)
    # reference basis function values
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=0)
           for i in range(evaluation_points.shape[0])]
    reference_basis = np.vstack(tmp).T
    # reference basis function first derivatives
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=1)
           for i in range(evaluation_points.shape[0])]
    reference_basis_derivatives = np.vstack(tmp).T
    # store all data and return
    reference_data = {'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'quadrature_weights': quadrature_weights,
                      'deg': deg,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives
    }
    return reference_data

#create finite element space
def create_fe_space(deg, reg, mesh):
    def bezier_extraction(knt, deg):
        # breakpoints
        brk = np.unique(knt)
        # number of elements
        nel = brk.shape[0]-1
        # number of knots
        m = knt.shape[0]
        # assuming an open knotvector, knt[a] is the last repetition of the first knot
        a = deg
        # next knot
        b = a+1
        # Bezier element being processed
        nb = 0
        # first extraction matrix
        C = [np.eye(deg+1,deg+1, dtype=float)]
        # this is where knot-insertion coefficients are saved
        alphas = np.zeros((np.maximum(deg-1,0),),dtype=float)
        while b < m:
            # initialize extraction matrix for next element
            C.append(np.eye(deg+1,deg+1))
            # save index of current knot
            i = b
            # find last occurence of current knot
            while b < m-1 and knt[b+1] == knt[b]:
                b += 1
            # multiplicity of current knot
            mult = b-i+1
            # if multiplicity is < deg, smoothness is at least C0 and extraction may differ from an identity matrix
            if mult < deg:
                numer = knt[b] - knt[a]
                # smoothness of splines
                r = deg - mult
                # compute linear combination coefficients
                for j in range(deg-1,mult-1,-1):
                    alphas[j-mult] = numer / (knt[a+j+1]-knt[a])
                for j in range(r):
                    s = mult+j
                    for k in range(deg,s,-1):
                        alpha = alphas[k-s-1]
                        C[nb][:,k] = alpha*C[nb][:,k] + (1.0-alpha)*C[nb][:,k-1]
                    save = r-j
                    if b < m:
                        C[nb+1][save-1:j+save+1,save-1] = C[nb][deg-j-1:deg+1,deg]
            # increment element index
            nb += 1
            if b < m:
                a = b
                b += 1
            C = C[:nel]
        return C
    # number of mesh elements
    nel = mesh['m']

    # unique breakpoints
    if nel == 1:
        brk = mesh['elements'].T[0]
    else:
        brk = np.concatenate((mesh['elements'][0],
                              np.array([mesh['elements'][1][-1]])), axis=0)
    # multiplicity of each breakpoint
    mult = deg - reg
    # knot vector for B-spline definition
    knt = np.concatenate((np.ones((deg+1,), dtype=float) * brk[0],
                          np.ones((deg+1,), dtype=float) * brk[-1],
                          np.repeat(brk[1:-1],mult)), axis=0)
    knt = np.sort(knt)
    # coefficients of linear combination
    C = bezier_extraction(knt, deg)
    # dimension of finite element space
    dim = knt.shape[0]-deg-1
    # connectivity information (i.e., which bases are non-zero on which element)
    econn = np.zeros((nel,deg+1), dtype=int)
    for i in range(nel):
        if i == 0:
            econn[i] = np.arange( deg+1)
        else:
            econn[i] = econn[i-1] + mult
    # save and return
    space = {'n': dim,
             'supported_bases': econn,
             'extraction_coefficients': C
    }
    return space

#Create a uniform mesh
def create_mesh(brk):
    elt = np.vstack((brk[0:-1], brk[1:]))
    mesh = {'elements': elt,
            'm': brk.shape[0]-1}
    return mesh

def create_param_map(mesh):
    # function that scales reference domain to mesh element
    def map(xi, x0, x1):
        return x0 + xi*(x1-x0)
    # derivative of the above map
    map_derivatives = mesh['elements'][1]-mesh['elements'][0]
    # derivative of the inverse of the above map
    imap_derivatives = 1./map_derivatives
    # store and return
    param_map = {'map': map,
                 'map_derivatives': map_derivatives,
                 'imap_derivatives': imap_derivatives
    }
    return param_map

def problem_A(z,Nj,dNj,Nk,dNk):
    return Nj*Nk

def problem_B(z,Nj,dNj,Nk,dNk):
    return dNj*dNk

def problem_C(z,Nj,dNj,Nk,dNk):
    return Nj*dNk

def problem_Dz(z,Nj,dNj,Nk,dNk):
    return Nj*dNk

def problem_rhs(z,Nj,dNj):
    return Nj

def problem_drhs(z,Nj,dNj):
    return dNj

#BC: P = F_zz, at surface z=0
def f(t, Fc, beta):
    if t < np.pi:
        return (1.0-beta)*Fc*(1.0-np.cos(t))
    else:
        return (1.0-beta)*Fc*2.0

#BC: P = F_zz, at surface z=0
def f_lab(gamma_w=10**4, T=9, H=3.5, Nc = 10, D = 0):
    #T [s] wave period
    #H [m] wave height
    #Nc #number of waves
    #D #[m] water depth
    def fnc(t):
        return 0.5 * gamma_w * H * math.sin(2*math.pi*(t/T)) + gamma_w*D
    
    return fnc

#BC: P = f, at surface z=0
def f_boat(dt, t, P1): #np.arrays
    time = np.arange(t[0], t[1]+dt, dt)
    P = np.linspace(P1[0], P1[1], len(time))
    for i in range(1,len(P1)-1):
        time_add = np.arange(t[i], t[i+1]+dt, dt)
        time = np.hstack((time,time_add[1:]))
        P = np.hstack((P,np.linspace(P1[i], P1[i+1], len(time_add))[1:]))
    time = np.array(time) 
    P = np.array(P)
    return time, P

#Assemble matrices for FEM with 1 layer of soil        
def assemble_fe_problem(mesh, space, ref_data, param_map):
    # retrieve data
    nel = mesh['m']
    n = space['n']
    RB = ref_data['reference_basis']
    dRB = ref_data['reference_basis_derivatives']
    xi = ref_data['evaluation_points']
    w = ref_data['quadrature_weights']
    deg = ref_data['deg']
    # element-wise assembly loop
    A = np.zeros((n,n))
    B = np.zeros((n,n))
    C = np.zeros((n,n))                                 
    for i in range(nel):
        # geometry information
        x = param_map['map'](xi,mesh['elements'][0][i],mesh['elements'][1][i])
        dx = param_map['map_derivatives'][i]
        dxi = param_map['imap_derivatives'][i]
        
        # extraction coefficients and FE basis
        E = space['extraction_coefficients'][i]
        N = np.matmul(E,RB) #ref
        dN = np.matmul(E,dRB)*dxi #ref
        
        # global indices of FE basis
        I = space['supported_bases'][i]
        # compute local integrals and add to global matrix
        for j in range(deg+1):
            I_j = I[j]
            for k in range(deg+1):
                I_k = I[k]
                tmp = problem_A(x,N[j],dN[j],N[k],dN[k])
                A[I_j,I_k] += dx*np.dot(w,tmp) #mesh
                tmp = problem_B(x,N[j],dN[j],N[k],dN[k])
                B[I_j,I_k] += dx*np.dot(w,tmp) #mesh
                tmp = problem_C(x,N[j],dN[j],N[k],dN[k])
                C[I_j,I_k] += dx*np.dot(w,tmp) #mesh
    # incorporate boundary conditions
    return A, B, C#, Mrhs_z

#Assemble matrices for FEM with 2 layers of soil
def assemble_fe_problem_2types(mesh, space, ref_data, param_map, c, d):
    # retrieve data
    nel = mesh['m']
    n = space['n']
    RB = ref_data['reference_basis']
    dRB = ref_data['reference_basis_derivatives']
    xi = ref_data['evaluation_points']
    w = ref_data['quadrature_weights']
    deg = ref_data['deg']
    # element-wise assembly loop
    A = np.zeros((n,n))
    dA_eps = np.zeros((n,n))
    dA_P = np.zeros((n,n))
    B = np.zeros((n,n))
    dB_uz = np.zeros((n,n))
    C = np.zeros((n,n))
    dC_uz = np.zeros((n,n))
    for i in range(nel):
        # geometry information
        x = param_map['map'](xi,mesh['elements'][0][i],mesh['elements'][1][i])
        dx = param_map['map_derivatives'][i]
        dxi = param_map['imap_derivatives'][i]
        
        # extraction coefficients and FE basis
        E = space['extraction_coefficients'][i]
        N = np.matmul(E,RB) #ref
        dN = np.matmul(E,dRB)*dxi #ref
        
        # global indices of FE basis
        I = space['supported_bases'][i]
        # compute local integrals and add to global matrix
        if i < nel - 101: #bottom 
            alpha_uz = d[0][0]
            alpha_P = d[1][0]
            a = c[0]
        else:
            alpha_uz = d[0][1]
            alpha_P = d[1][1]
            a = c[1]
        for j in range(deg+1):
            I_j = I[j]
            for k in range(deg+1):
                I_k = I[k]
                tmp = problem_A(x,N[j],dN[j],N[k],dN[k])
                A[I_j,I_k] += dx*np.dot(w,tmp) #mesh
                dA_eps[I_j,I_k] += dx*np.dot(w,tmp) * alpha_uz
                dA_P[I_j,I_k] += dx*np.dot(w,tmp) * alpha_P
                
                tmp = problem_B(x,N[j],dN[j],N[k],dN[k])
                B[I_j,I_k] += dx*np.dot(w,tmp) #mesh
                dB_uz[I_j,I_k] += dx*np.dot(w,tmp)*a
                
                tmp = problem_C(x,N[j],dN[j],N[k],dN[k])
                C[I_j,I_k] += dx*np.dot(w,tmp) #mesh
                dC_uz[I_j,I_k] += dx*np.dot(w,tmp) * alpha_uz
    # incorporate boundary conditions
    return A, dA_eps, dA_P, B, dB_uz, C, dC_uz


#c = sum c_j N_j, 1 layer of soil    
def sum_coeff(mesh, space ,ref_data, param_map, coeff):
    # retrieve data
    n = space['n']
    nel = mesh['m']
    RB = ref_data['reference_basis']
    dRB = ref_data['reference_basis_derivatives']
    xi = ref_data['evaluation_points']
    deg = ref_data['deg']
    #initialize data
    step = np.shape(RB)[1]
    num_t = np.shape(coeff)[1]
    coeff_sol = np.zeros((4,num_t,nel,step))
    dcoeff_sol = np.zeros((4,num_t,nel,step))
    z_lst = np.zeros((num_t, nel, step))
    for t in range(num_t): 
        for i in range(nel):
            # geometry information
            z = param_map['map'](xi,mesh['elements'][0][i],mesh['elements'][1][i])
            dxi = param_map['imap_derivatives'][i]
            E = space['extraction_coefficients'][i]
            N = np.matmul(E,RB)
            dN = dxi*np.matmul(E,dRB)
            I = space['supported_bases'][i]
            for j in range(num_var):
                coeff_sol[j,t,i] = np.matmul(coeff[j,t,I],N)
                dcoeff_sol[j,t,i] = np.matmul(coeff[j,t,I],dN)

            z_lst[t,i] = z
    return coeff_sol, dcoeff_sol, z_lst

#c = sum c_j N_j, 2 layers of soil
def sum_coeff_2types(mesh, space ,ref_data, param_map, coeff, const):
    # retrieve data
    n = space['n']
    nel = mesh['m']
    RB = ref_data['reference_basis']
    dRB = ref_data['reference_basis_derivatives']
    xi = ref_data['evaluation_points']
    deg = ref_data['deg']
    #initialize data
    step = np.shape(RB)[1]
    num_t = np.shape(coeff)[1]
    num_var = np.shape(coeff)[0]
    coeff_sol = np.zeros((num_var,num_t,nel,step))
    dcoeff_sol = np.zeros((num_var,num_t,nel,step))
    z_lst = np.zeros((num_t, nel, step))
    for t in range(num_t): 
        for i in range(nel):
            # geometry information
            z = param_map['map'](xi,mesh['elements'][0][i],mesh['elements'][1][i])
            dxi = param_map['imap_derivatives'][i]
            E = space['extraction_coefficients'][i]
            N = np.matmul(E,RB)
            dN = dxi*np.matmul(E,dRB)
            I = space['supported_bases'][i]
            if i < nel-101: #bottom
                c = const[0]
            else:
                c = const[1]
            for j in range(num_var):
                coeff_sol[j,t,i] = np.matmul(coeff[j,t,I],N)
                dcoeff_sol[j,t,i] = np.matmul(coeff[j,t,I],dN)
            coeff_sol[0,t,i] = -c*coeff_sol[0,t,i]
            dcoeff_sol[0,t,i] = -c*dcoeff_sol[0,t,i]

            z_lst[t,i] = z
        
    return coeff_sol, dcoeff_sol, z_lst

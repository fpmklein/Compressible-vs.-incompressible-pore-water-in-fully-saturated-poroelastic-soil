import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data1d_paper import data
matplotlib.rc('font', size=26)

#plot solutions for effective stress, eps, P, u_z and their derivatives at different times
def plot(coeff, dcoeff, z_lst, time, compr = "Incompressible"):
    fig, ax = plt.subplots(2,2,layout="constrained")
    fig2, ax2 = plt.subplots(2,2,layout="constrained")
    # colors
    col_stress = ['darkgreen', 'olivedrab','yellowgreen', 'lightgreen', 'mediumspringgreen']
    col_eps = ['darkslategrey', 'darkcyan', 'cyan', 'paleturquoise', 'lightcyan']
    col_P = ['midnightblue','mediumblue','indigo', 'darkorchid', 'orchid']
    col_u = ['darkmagenta', 'magenta', 'deeppink', 'palevioletred', 'pink']
    ncol=len(col_stress)
    nel = np.shape(coeff)[2]
    for t in range(np.shape(coeff)[1]):
        stop_time = time[t]
        label = True
        for i in range(nel):
            effStress_zz = coeff[0,t,i]
            eps = coeff[1,t,i]
            P = coeff[2,t,i]
            u_z = coeff[3,t,i]

            deffStress_zz = dcoeff[0,t,i]
            deps = dcoeff[1,t,i]
            dP = dcoeff[2,t,i]
            du_z = dcoeff[3,t,i]
            z = z_lst[t,i]
            #col[np.mod(t,ncol)]
            if label == True:
                ax[0][0].plot(z,effStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax[0][1].plot(z,eps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax[1][0].plot(z,P,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax[1][1].plot(z,u_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                
                ax2[0][0].plot(z,deffStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax2[0][1].plot(z,deps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax2[1][0].plot(z,dP,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax2[1][1].plot(z,du_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                label = False
            else:
                ax[0][0].plot(z,effStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax[0][1].plot(z,eps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax[1][0].plot(z,P,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax[1][1].plot(z,u_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3)
                
                ax2[0][0].plot(z,deffStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax2[0][1].plot(z,deps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax2[1][0].plot(z,dP,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax2[1][1].plot(z,du_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3)
                
            if i == 0:
                print("deps(-L)/dz = ", deps[0])
                print("dP(-L)/dz = ", dP[0])
                print("uz(-L) = ", u_z[0])
            elif i == nel-1:
                print("P(0) = ", P[-1])        
            
    ax[0][0].set_title(r"Effective stress",fontsize=16)# + f' at time {stop_time} s')
    ax[0][1].set_title(r"Volumetric Strain",fontsize=16)#+ f' at time {stop_time} s')
    ax[1][0].set_title(r"Water Pressure",fontsize=16)#+ f' at time {stop_time} s')
    ax[1][1].set_title(r"$z$-displacement",fontsize=16)#+ f' at time {stop_time} s')
    ax[0][0].set_ylabel(r"$\sigma'_{zz}$",fontsize=12)
    ax[0][1].set_ylabel(r"$\epsilon_{vol}$",fontsize=12)
    ax[1][0].set_ylabel(r"$P$",fontsize=12)
    ax[1][1].set_ylabel(r"$u_z$",fontsize=12)
    ax[0][0].set_xlabel(r"$z$",fontsize=12)
    ax[0][1].set_xlabel(r"$z$",fontsize=12)
    ax[1][0].set_xlabel(r"$z$",fontsize=12)
    ax[1][1].set_xlabel(r"$z$",fontsize=12)
    ax[0][0].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)
    ax[1][1].grid(True)
    ax[0][0].legend(loc='upper right')
    ax[0][1].legend(loc='upper right')
    ax[1][0].legend(loc='upper right')
    ax[1][1].legend(loc='upper right')

    ax2[0][0].set_title(r"Derivative of Effective Stress wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[0][1].set_title(r"Derivative of Volumetric Strain wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[1][0].set_title(r"Derivative of Water Pressure wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[1][1].set_title(r"Derivative of $z$-displacement wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[0][0].set_ylabel(r"$\frac{\partial \sigma'_{zz}}{\partial z}$",fontsize=12)
    ax2[0][1].set_ylabel(r"$\frac{\partial \epsilon_{vol}}{\partial z}$",fontsize=12)
    ax2[1][0].set_ylabel(r"$\frac{\partial P}{\partial z}$",fontsize=12)
    ax2[1][1].set_ylabel(r"$\frac{\partial u_z}{\partial z}$",fontsize=12)
    ax2[0][0].set_xlabel(r"$z$",fontsize=12)
    ax2[0][1].set_xlabel(r"$z$",fontsize=12)
    ax2[1][0].set_xlabel(r"$z$",fontsize=12)
    ax2[1][1].set_xlabel(r"$z$",fontsize=12)
    ax2[0][0].grid(True)
    ax2[0][1].grid(True)
    ax2[1][0].grid(True)
    ax2[1][1].grid(True)
    ax2[0][0].legend(loc='upper right')
    ax2[0][1].legend(loc='upper right')
    ax2[1][0].legend(loc='upper right')
    ax2[1][1].legend(loc='upper right')
    fig.suptitle(r"Numerical solution at different times" + "\n " + " incompressible water", fontsize=20) #for $\sigma'_{zz}, \epsilon_{vol}, P, u_z$
    fig2.suptitle(r"Numerical solution of derivatives wrt $z$ at different times"+ "\n " + " incompressible water", fontsize=20) #of $\sigma'_{zz}, \epsilon_{vol}, P, u_z$
    #fig.tight_layout()
    #fig2.tight_layout()
    plt.show()

#plot normalised solutions for effective stress, eps, P, u_z and their derivatives at different times
def plot_norm(coeff, dcoeff, z_lst, Z, time, gamma_w, compr = "Incompressible"):
    fig, ax = plt.subplots(2,2,layout="constrained")
    fig2, ax2 = plt.subplots(2,2,layout="constrained")
    # colors
    col_stress = ['darkgreen', 'olivedrab','yellowgreen', 'lightgreen', 'mediumspringgreen']
    col_eps = ['darkslategrey', 'darkcyan', 'cyan', 'paleturquoise', 'lightcyan']
    col_P = ['midnightblue','mediumblue','indigo', 'darkorchid', 'orchid']
    col_u = ['darkmagenta', 'magenta', 'deeppink', 'palevioletred', 'pink']
    ncol=len(col_stress)
    nel = np.shape(coeff)[2]
    peak_pressure = 17500 #[N/m^2]
    length_z = -Z #[m]
    for t in range(np.shape(coeff)[1]):
        stop_time = time[t]
        label = True
        for i in range(nel):
            effStress_zz = coeff[0,t,i] / peak_pressure
            eps = coeff[1,t,i] / peak_pressure
            P = coeff[2,t,i] / peak_pressure
            u_z = coeff[3,t,i] / peak_pressure

            deffStress_zz = dcoeff[0,t,i] / peak_pressure
            deps = dcoeff[1,t,i] / peak_pressure
            dP = dcoeff[2,t,i] / peak_pressure
            du_z = dcoeff[3,t,i] / peak_pressure
            z = z_lst[t,i] / length_z
            #col[np.mod(t,ncol)]
            if label == True:
                ax[0][0].plot(z,effStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax[0][1].plot(z,eps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax[1][0].plot(z,P,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax[1][1].plot(z,u_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                
                ax2[0][0].plot(z,deffStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax2[0][1].plot(z,deps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax2[1][0].plot(z,dP,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                ax2[1][1].plot(z,du_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3, label = f't = {stop_time} s')
                label = False
            else:
                ax[0][0].plot(z,effStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax[0][1].plot(z,eps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax[1][0].plot(z,P,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax[1][1].plot(z,u_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3)
                
                ax2[0][0].plot(z,deffStress_zz,color = col_stress[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax2[0][1].plot(z,deps,color = col_eps[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax2[1][0].plot(z,dP,color = col_P[np.mod(t,ncol)], linestyle = '-', lw=3)
                ax2[1][1].plot(z,du_z,color = col_u[np.mod(t,ncol)], linestyle = '-', lw=3)
                
            if i == 0:
                print("deps(-L)/dz = ", deps[0])
                print("dP(-L)/dz = ", dP[0])
                print("uz(-L) = ", u_z[0])
            elif i == nel-1:
                print("P(0) = ", P[-1])        
            
    ax[0][0].set_title(r"Effective stress",fontsize=16)# + f' at time {stop_time} s')
    ax[0][1].set_title(r"Volumetric Strain",fontsize=16)#+ f' at time {stop_time} s')
    ax[1][0].set_title(r"Water Pressure",fontsize=16)#+ f' at time {stop_time} s')
    ax[1][1].set_title(r"$z$-displacement",fontsize=16)#+ f' at time {stop_time} s')
    ax[0][0].set_ylabel(r"${\sigma'_{zz}} / |P|$",fontsize=12)
    ax[0][1].set_ylabel(r"$\epsilon_{vol} / |P|$",fontsize=12)
    ax[1][0].set_ylabel(r"$P / |P|$",fontsize=12)
    ax[1][1].set_ylabel(r"$u_z / |P|$",fontsize=12)
    ax[0][0].set_xlabel(r"$z / |z|$",fontsize=12)
    ax[0][1].set_xlabel(r"$z / |z|$",fontsize=12)
    ax[1][0].set_xlabel(r"$z / |z|$",fontsize=12)
    ax[1][1].set_xlabel(r"$z / |z|$",fontsize=12)
    ax[0][0].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)
    ax[1][1].grid(True)
    ax[0][0].legend(loc='upper right')
    ax[0][1].legend(loc='upper right')
    ax[1][0].legend(loc='upper right')
    ax[1][1].legend(loc='upper right')

    ax2[0][0].set_title(r"Derivative of Effective Stress wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[0][1].set_title(r"Derivative of Volumetric Strain wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[1][0].set_title(r"Derivative of Water Pressure wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[1][1].set_title(r"Derivative of $z$-displacement wrt $z$",fontsize=16)#+ f' at time {stop_time} s')
    ax2[0][0].set_ylabel(r"$\frac{\partial \sigma'_{zz}}{\partial z} / |P|$",fontsize=12)
    ax2[0][1].set_ylabel(r"$\frac{\partial \epsilon_{vol}}{\partial z} / |P|$",fontsize=12)
    ax2[1][0].set_ylabel(r"$\frac{\partial P}{\partial z} / |P|$",fontsize=12)
    ax2[1][1].set_ylabel(r"$\frac{\partial u_z}{\partial z} / |P|$",fontsize=12)
    ax2[0][0].set_xlabel(r"$z / |z|$",fontsize=12)
    ax2[0][1].set_xlabel(r"$z / |z|$",fontsize=12)
    ax2[1][0].set_xlabel(r"$z / |z|$",fontsize=12)
    ax2[1][1].set_xlabel(r"$z / |z|$",fontsize=12)
    ax2[0][0].grid(True)
    ax2[0][1].grid(True)
    ax2[1][0].grid(True)
    ax2[1][1].grid(True)
    ax2[0][0].legend(loc='upper right')
    ax2[0][1].legend(loc='upper right')
    ax2[1][0].legend(loc='upper right')
    ax2[1][1].legend(loc='upper right')
    fig.suptitle(r"Normalised numerical solution at different times" + "\n " + compr + " water", fontsize=20) #for $\sigma'_{zz}, \epsilon_{vol}, P, u_z$
    fig2.suptitle(r"Normalised numerical solution of derivatives wrt $z$ at different times"+ "\n " + compr + " water", fontsize=20) #of $\sigma'_{zz}, \epsilon_{vol}, P, u_z$
    #fig.tight_layout()
    #fig2.tight_layout()
    plt.show()

#plot normalised solutions for P with data against space
def plot_pressure_norm_data(x, coeff, time, gamma_w, H, Z, data_part, data_plot, label1, label2): #coeff of pressure
    fig, ax = plt.subplots(1,1,layout="constrained")
    
    # retrieve data
    peak_pressure = 0.5 * H * gamma_w #+ 52000 #[N/m^2]
    length_x = Z #[m] -1.8
    stop_time = time
    color_lst = ['midnightblue', 'purple']
    label_lst = [label1, label2]
    x_lst = x.flatten()
    x_max = max(x_lst)
    x_min = min(x_lst)
    
    x = x/(x_max-x_min)

    color_data = ['lightblue', 'pink']
    for par in range(np.shape(coeff)[0]):
        c = coeff[par]
##        lst = c.flatten()
##        max_var = max(lst)
##        min_var = min(lst)
        var = c/peak_pressure #(0.5*(max_var - min_var))#(c - min_var)/(max_var - min_var)
        ax.set_title(f"Normalised water pressure with real data at t = {stop_time} s", fontsize=35)        
        ax.set_xlabel(r"$|P| / P_0 $",fontsize=30)
        #ax.plot(z,var,color = 'midnightblue', linestyle = '-', lw=3)
        v = ax.plot(var,x,color = color_lst[par], linestyle = '-', lw=6)
        print('shape v = ', np.shape(np.array(v)))
        v[0].set_label(label_lst[par])            
    z, data_a, data_b, data_c, data_d, xticks = data(data_part)            
    if data_plot == 'a':
        data_lst = data_a
        for i in range(np.shape(data_lst)[0]):
            ax.scatter(data_lst[i,:], z,  color = color_data[i], lw=6, label = 'Data ' + label_lst[i])
    elif data_plot == 'b':
        data_lst = data_b
        for i in range(np.shape(data_lst)[0]):
            ax.scatter(data_lst[i,:], z,  color = color_data[i], lw=6, label = 'Data ' + label_lst[i])
    elif data_plot == 'c':
        data_lst = data_c
        for i in range(np.shape(data_lst)[0]):
            ax.scatter(data_lst[i,:], z,  color = color_data[i], lw=6, label = 'Data ' + label_lst[i])
    elif data_plot == 'd':
        data_lst = data_d
        for i in range(np.shape(data_lst)[0]):
            ax.scatter(data_lst[i,:], z,  color = color_data[i], lw=6, label = 'Data ' + label_lst[i])
    else:
        data_lst = data_a
        ax.scatter(data_lst, z,  color = color_data[0], lw=6, label = 'Data ' + label_lst[0])
        

    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26) 
    ax.set_xlim([0,1])
    ax.set_ylim([-1.0,0])
    ax.set_ylabel(r"$z/h$",fontsize=30)
    ax.grid(True)
    plt.legend(loc='upper left',fontsize = 22)
    plt.show()


def plot_pressure_data(time, coeff, gamma_w=10**4, data_name="", date=""):
    fig, ax = plt.subplots(1,1,layout="constrained")
    
    # retrieve data
    color_lst1 = ['royalblue', 'gold', 'darkorange']
    color_lst2 = ['midnightblue', 'darkgoldenrod', 'red']
    label_lst = ['P1', 'P2', 'P4']
    t_lst, P1_lst, P2_lst, P3_lst, P4_lst, xticks = data(data_name)
    P_lst = np.array([P1_lst, P2_lst, P4_lst])
    P0_lst = np.array([P1_lst[0], P2_lst[0], P4_lst[0]])
    time_pos, time_lab = list(map(int,xticks[:,0])), xticks[:,1]
    print('shape time, t_lst = ', np.shape(time), np.shape(t_lst))
    for i in range(np.shape(coeff)[0]):
        c = coeff[i] + P0_lst[i]*gamma_w
        ax.set_title(f"Water pressure with real data " + date, fontsize=35)        
        ax.set_xlabel(r"$t$ [HH:MM:SS]",fontsize=30)
        v = ax.plot(time, c, color = color_lst2[i], linestyle = '-', lw=5)
        v[0].set_label(label_lst[i])
        
    for i in range(np.shape(coeff)[0]):
        P = P_lst[i]*gamma_w #+ (P0_lst[i]-P0_lst[0])
        v2 = ax.plot(t_lst, P, color = color_lst1[i], linestyle = '--', lw=5)
        v2[0].set_label('Data ' + label_lst[i])

    plt.xticks(time_pos, time_lab, fontsize = 26)
    #plt.xticks(time[0::70], convert_sec_timer(time[0::70]), fontsize = 26)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.yticks(fontsize = 26)

    ax.set_ylim(gamma_w*np.array([3.3,4.0]))
    ax.set_ylabel(r"$P$ [Pa]",fontsize=30)
    ax.grid(True)
    #ax.legend(loc='upper left')
    plt.legend(loc='upper left',fontsize = 22, ncol=2)
    plt.show()

#plot solutions for effecitve stress, eps, u_z with data against time
def plot_others_time(time, coeff, gamma_w, lm, Z, data_name="", date="", model=0):
    fig1, ax1 = plt.subplots(1,1,layout="constrained")
    fig2, ax2 = plt.subplots(1,1,layout="constrained")
    fig3, ax3 = plt.subplots(1,1,layout="constrained")
    fig4, ax4 = plt.subplots(1,1,layout="constrained")
    color_lst1 = ['royalblue', 'gold', 'darkorange']
    color_lst2 = ['midnightblue', 'darkgoldenrod', 'red']
    label_lst = ['P1', 'P2', 'P4']
    t_lst, P1_lst, P2_lst, P3_lst, P4_lst, xticks = data(data_name)
    P0_lst = np.array([P1_lst[0], P2_lst[0], P4_lst[0]])
    time_pos, time_lab = list(map(int,xticks[:,0])), xticks[:,1]
    steady_state = np.array([P0_lst, np.zeros(np.shape(P0_lst))])
    print('shape time, t_lst = ', np.shape(time), np.shape(t_lst))
    for i in range(np.shape(coeff)[0]):
        c_eff = coeff[i,0]/gamma_w - steady_state[model,i]
        c_eps = coeff[i,1] + gamma_w*steady_state[model,i] / lm
        c_uz = coeff[i,3] + (1+Z)*gamma_w*steady_state[model,i] / lm
        #c_vol_bal = coeff[i,4]

        v1 = ax1.plot(time, c_eff, color = color_lst2[i], linestyle = '-', lw=5)
        v2 = ax2.plot(time, c_eps, color = color_lst2[i], linestyle = '-', lw=5)
        v3 = ax3.plot(time, c_uz, color = color_lst2[i], linestyle = '-', lw=5)
        v1[0].set_label(label_lst[i])
        v2[0].set_label(label_lst[i])
        v3[0].set_label(label_lst[i])

    ax1.set_title(f"Effective stress " + date, fontsize=35)
    ax2.set_title(f"Volumetric strain " + date, fontsize=35)
    ax3.set_title(r"$z$-displacement " + date, fontsize=35) 
    ax1.set_xticks(time_pos, time_lab, fontsize = 26)
    ax2.set_xticks(time_pos, time_lab, fontsize = 26)
    ax3.set_xticks(time_pos, time_lab, fontsize = 26)
    #ax1.set_ylim([-4.0, -3.3])
    #ax2.set_ylim([0.045, 0.055])
    #ax3.set_ylim([0.151, 0.161])
    ax1.tick_params(axis='y',labelsize = 26)
    ax2.tick_params(axis='y',labelsize = 26)
    ax3.tick_params(axis='y',labelsize = 26)
    ax1.set_xlabel(r"$t [s]$",fontsize=30)
    ax2.set_xlabel(r"$t [s]$",fontsize=30)
    ax3.set_xlabel(r"$t [s]$",fontsize=30)
    ax1.set_ylabel(r"$\sigma'_{zz} [m]$",fontsize=30)
    ax2.set_ylabel(r"$\epsilon_{vol} [-]$",fontsize=30)
    ax3.set_ylabel(r"$u_z [m]$",fontsize=30)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.legend(loc='upper left',fontsize = 22)
    ax2.legend(loc='upper left',fontsize = 22)
    ax3.legend(loc='upper left',fontsize = 22)
    plt.show()

#plot derivative wrt z solutions for effecitve stress, eps, u_z with data against time
def plot_others_time_der(time, dcoeff, gamma_w, lm, Z, data_name="", date="", model=0):    
    fig, ax = plt.subplots(1,1,layout="constrained")
    fig1, ax1 = plt.subplots(1,1,layout="constrained")
    fig2, ax2 = plt.subplots(1,1,layout="constrained")
    fig3, ax3 = plt.subplots(1,1,layout="constrained")
    color_lst1 = ['royalblue', 'gold', 'darkorange']
    color_lst2 = ['midnightblue', 'darkgoldenrod', 'red']
    label_lst = ['P1', 'P2', 'P4']
    t_lst, P1_lst, P2_lst, P3_lst, P4_lst, xticks = data(data_name)
    time_pos, time_lab = list(map(int,xticks[:,0])), xticks[:,1]
    for i in range(np.shape(dcoeff)[0]):
        c_eff = dcoeff[i,0]/gamma_w
        c_eps = dcoeff[i,1]
        c_P = dcoeff[i,2]/gamma_w
        c_uz = dcoeff[i,3]

        v = ax.plot(time, c_P, color = color_lst2[i], linestyle = '-', lw=5)
        v1 = ax1.plot(time, c_eff, color = color_lst2[i], linestyle = '-', lw=5)
        v2 = ax2.plot(time, c_eps, color = color_lst2[i], linestyle = '-', lw=5)
        v3 = ax3.plot(time, c_uz, color = color_lst2[i], linestyle = '-', lw=5)
        v[0].set_label(label_lst[i])
        v1[0].set_label(label_lst[i])
        v2[0].set_label(label_lst[i])
        v3[0].set_label(label_lst[i])

    ax.set_title(r"Derivative of water pressure wrt $z$ " + date, fontsize=35)
    ax1.set_title(r"Derivative of effective stress wrt $z$ " + date, fontsize=35)
    ax2.set_title(r"Derivative of volumetric strain wrt $z$ " + date, fontsize=35)
    ax3.set_title(r"Derivative of $z$-displacement wrt $z$ " + date, fontsize=35)
    ax.set_xticks(time_pos, time_lab, fontsize = 26)
    ax1.set_xticks(time_pos, time_lab, fontsize = 26)
    ax2.set_xticks(time_pos, time_lab, fontsize = 26)
    ax3.set_xticks(time_pos, time_lab, fontsize = 26)
##    ax.set_ylim([3.3,4.0])
##    ax1.set_ylim([-4.0, -3.3])
##    ax2.set_ylim([0.045, 0.055])
##    ax3.set_ylim([0.151, 0.161])
    ax.tick_params(axis='y',labelsize = 26)
    ax1.tick_params(axis='y',labelsize = 26)
    ax2.tick_params(axis='y',labelsize = 26)
    ax3.tick_params(axis='y',labelsize = 26)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel(r"$t$",fontsize=30)
    ax1.set_xlabel(r"$t$",fontsize=30)
    ax2.set_xlabel(r"$t$",fontsize=30)
    ax3.set_xlabel(r"$t$",fontsize=30)
    ax.set_ylabel(r"$\frac{\partial P}{\partial z}$",fontsize=30)
    ax1.set_ylabel(r"$\frac{\partial \sigma'_{zz}}{\partial z}$",fontsize=30)
    ax2.set_ylabel(r"$\frac{\partial \epsilon_{vol}}{\partial z}$",fontsize=30)
    ax3.set_ylabel(r"$\frac{\partial u_z}{\partial z}$",fontsize=30)
    ax.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax.legend(loc='upper left',fontsize = 22)
    ax1.legend(loc='upper left',fontsize = 22)
    ax2.legend(loc='upper left',fontsize = 22)
    ax3.legend(loc='upper left',fontsize = 22)
    plt.show()

#plot solutions for volume balance, using weak equation of eps
def plot_volume_balance(time, coeff, gamma_w=10**4, data_name="", date=""):
    fig, ax = plt.subplots(1,1,layout="constrained")
    
    # retrieve data
    color_lst1 = ['royalblue', 'gold', 'darkorange']
    color_lst2 = ['midnightblue', 'darkgoldenrod', 'red']
    label_lst = ['P1', 'P2', 'P4']
    t_lst, P1_lst, P2_lst, P3_lst, P4_lst, xticks = data(data_name)
    time_pos, time_lab = list(map(int,xticks[:,0])), xticks[:,1]
    print('shape time, t_lst = ', np.shape(time), np.shape(t_lst))
    for i in range(np.shape(coeff)[0]):
        c = coeff[i]
        ax.set_title(f"Volume balance " + date, fontsize=35)        
        ax.set_xlabel(r"$t $",fontsize=30)
        v = ax.plot(time, c, color = color_lst2[i], linestyle = '-', lw=5)
        v[0].set_label(label_lst[i])
        
    plt.xticks(time_pos, time_lab, fontsize = 26)
    #plt.xticks(time[0::70], convert_sec_timer(time[0::70]), fontsize = 26)
    plt.yticks(fontsize = 26)
    ax.set_ylabel(r"Volume balance",fontsize=30)
    ax.grid(True)
    #ax.legend(loc='upper left')
    plt.legend(loc='upper left',fontsize = 22, ncol=2)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=15)


#Heatmap plot solution for one variable and its derivative wrt x and z
def plot_2dsolution_heatmap_single_variable(coeff, dcoeff, x_lst, model, title, symb):
    #fig, ax = plt.subplots(1,1,layout="constrained", figsize=(17,10))
    fig = plt.figure(figsize=(17,10))
    ax = fig.add_subplot(1, 1, 1) 
    graph_omega = plt.scatter(x_lst[0], x_lst[1], c=coeff, s=0.01)
    ax.set_xlabel(r"$x$",fontsize=12)
    ax.set_ylabel(r"$z$",fontsize=12)
    ax.grid(True)
    ax.set_title(title,fontsize=20)
    plt.colorbar(graph_omega)
    fig.savefig(title+"_"+model+".png")
    print("single fig saved")
    plt.close()
    
    fig2, ax2 = plt.subplots(1,2,layout="constrained", figsize=(17,10))
    derX_graph_omega = ax2[0].scatter(x_lst[0], x_lst[1], c=dcoeff[0], s=0.01)
    derZ_graph_omega = ax2[1].scatter(x_lst[0], x_lst[1], c=dcoeff[1], s=0.01)
    ax2[0].set_xlabel(r"$x$",fontsize=12)
    ax2[1].set_xlabel(r"$x$",fontsize=12)
    ax2[0].set_ylabel(r"$z$",fontsize=12)
    ax2[1].set_ylabel(r"$z$",fontsize=12)
    ax2[0].set_title(r"Derivative of " + title + " wrt $x$",fontsize=20)
    ax2[1].set_title(r"Derivative of " + title + " wrt $z$",fontsize=20)    
    
    ax2[0].grid(True)
    ax2[1].grid(True)

    #plt.legend()
    plt.colorbar(derX_graph_omega)
    plt.colorbar(derZ_graph_omega)

    fig2.savefig("Derivatives_"+title+"_"+model+".png")
    print("derivatives of single fig saved")

    #plt.show()
    plt.close()

    
#Heatmap plot solutions for eps, P, u_x, u_z and their derivatives wrt x and z
def plot_2dsolution_heatmap(coeff, dcoeff, x_lst, model):
    fig, ax = plt.subplots(2,2,layout="constrained", figsize=(17,10))
    fig2, ax2 = plt.subplots(2,2,layout="constrained", figsize=(17,10))
    fig3, ax3 = plt.subplots(2,2,layout="constrained", figsize=(17,10))
    
    graph_eps = ax[0][0].scatter(x_lst[0], x_lst[1], c=coeff[1], s=0.01)
    graph_P = ax[1][0].scatter(x_lst[0], x_lst[1], c=coeff[2], s=0.01)
    graph_ux = ax[0][1].scatter(x_lst[0], x_lst[1], c=coeff[3], s=0.01)
    graph_uz = ax[1][1].scatter(x_lst[0], x_lst[1], c=coeff[4], s=0.01)

    derX_graph_eps = ax2[0][0].scatter(x_lst[0], x_lst[1], c=dcoeff[1,0], s=0.01)
    derX_graph_P = ax2[1][0].scatter(x_lst[0], x_lst[1], c=dcoeff[2,0], s=0.01)
    derX_graph_ux = ax2[0][1].scatter(x_lst[0], x_lst[1], c=dcoeff[3,0], s=0.01)
    derX_graph_uz = ax2[1][1].scatter(x_lst[0], x_lst[1], c=dcoeff[4,0], s=0.01)

    derZ_graph_eps = ax3[0][0].scatter(x_lst[0], x_lst[1], c=dcoeff[1,1], s=0.01)
    derZ_graph_P = ax3[1][0].scatter(x_lst[0], x_lst[1], c=dcoeff[2,1], s=0.01)
    derZ_graph_ux = ax3[0][1].scatter(x_lst[0], x_lst[1], c=dcoeff[3,1], s=0.01)
    derZ_graph_uz = ax3[1][1].scatter(x_lst[0], x_lst[1], c=dcoeff[4,1], s=0.01)
    
    f=19
    ax[0][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax[0][1].set_xlabel(r"$x$ [m]",fontsize=f)
    ax[1][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax[1][1].set_xlabel(r"$x$ [m]",fontsize=f)

    ax2[0][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax2[0][1].set_xlabel(r"$x$ [m]",fontsize=f)
    ax2[1][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax2[1][1].set_xlabel(r"$x$ [m]",fontsize=f)

    ax3[0][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax3[0][1].set_xlabel(r"$x$ [m]",fontsize=f)
    ax3[1][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax3[1][1].set_xlabel(r"$x$ [m]",fontsize=f)

    ax[0][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax[0][1].set_ylabel(r"$z$ [m]",fontsize=f)
    ax[1][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax[1][1].set_ylabel(r"$z$ [m]",fontsize=f)
    
    ax2[0][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax2[0][1].set_ylabel(r"$z$ [m]",fontsize=f)
    ax2[1][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax2[1][1].set_ylabel(r"$z$ [m]",fontsize=f)

    ax3[0][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax3[0][1].set_ylabel(r"$z$ [m]",fontsize=f)
    ax3[1][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax3[1][1].set_ylabel(r"$z$ [m]",fontsize=f)

    f2 = 23
    ax[0][0].set_title("Volumetric strain",fontsize=f2)
    ax[1][0].set_title("Water pressure",fontsize=f2)
    ax[0][1].set_title(r"$x$-displacement",fontsize=f2)
    ax[1][1].set_title(r"$z$-displacement",fontsize=f2)

    ax2[0][0].set_title(r"Derivative of volumetric strain wrt $x$",fontsize=f2)
    ax2[1][0].set_title(r"Derivative of water pressure wrt $x$",fontsize=f2)
    ax2[0][1].set_title(r"Derivative of $x$-displacement wrt $x$",fontsize=f2)
    ax2[1][1].set_title(r"Derivative of $z$-displacement wrt $x$",fontsize=f2)

    ax3[0][0].set_title(r"Derivative of volumetric strain wrt $z$",fontsize=f2)
    ax3[1][0].set_title(r"Derivative of water pressure wrt $z$",fontsize=f2)
    ax3[0][1].set_title(r"Derivative of $x$-displacement wrt $z$",fontsize=f2)
    ax3[1][1].set_title(r"Derivative of $z$-displacement wrt $z$",fontsize=f2)
    
    ax[0][0].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)
    ax[1][1].grid(True)

    ax2[0][0].grid(True)
    ax2[0][1].grid(True)
    ax2[1][0].grid(True)
    ax2[1][1].grid(True)

    ax3[0][0].grid(True)
    ax3[0][1].grid(True)
    ax3[1][0].grid(True)
    ax3[1][1].grid(True)

    #plt.legend()

    plt.colorbar(graph_eps)
    plt.colorbar(graph_P)
    plt.colorbar(graph_ux)
    plt.colorbar(graph_uz)

    plt.colorbar(derX_graph_eps)
    plt.colorbar(derX_graph_P)
    plt.colorbar(derX_graph_ux)
    plt.colorbar(derX_graph_uz)
    
    plt.colorbar(derZ_graph_eps)
    plt.colorbar(derZ_graph_P)
    plt.colorbar(derZ_graph_ux)
    plt.colorbar(derZ_graph_uz)

    fig.savefig("Solutions_"+model+".png")
    print("fig saved")
    fig2.savefig("Solutions_"+model+"_der_x.png")
    print("fig2 saved")
    fig3.savefig("Solutions_"+model+"_der_z.png")
    print("fig3 saved")

    #plt.show()
    plt.close()

#Heatmap plot solutions for omega, effective stress, volume balance and normal stress
def plot_2dsolution_heatmap_rest(coeff, dcoeff, x_lst, model):
    fig, ax = plt.subplots(2,2,layout="constrained", figsize=(17,10))
    
    graph_omega = ax[0][0].scatter(x_lst[0], x_lst[1], c=coeff[5], s=0.01)
    graph_vol_balance = ax[1][0].scatter(x_lst[0], x_lst[1], c=coeff[7], s=0.01)
    graph_eff_stress = ax[0][1].scatter(x_lst[0], x_lst[1], c=coeff[0], s=0.01)
    graph_shear_stress = ax[1][1].scatter(x_lst[0], x_lst[1], c=coeff[6], s=0.01)

    f=19
    ax[0][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax[0][1].set_xlabel(r"$x$ [m]",fontsize=f)
    ax[1][0].set_xlabel(r"$x$ [m]",fontsize=f)
    ax[1][1].set_xlabel(r"$x$ [m]",fontsize=f)

    ax[0][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax[0][1].set_ylabel(r"$z$ [m]",fontsize=f)
    ax[1][0].set_ylabel(r"$z$ [m]",fontsize=f)
    ax[1][1].set_ylabel(r"$z$ [m]",fontsize=f)

    f2=23
    ax[0][0].set_title(r"Vorticity ($\omega$)",fontsize=f2)
    ax[1][0].set_title("Volume balance",fontsize=f2)
    ax[0][1].set_title(r"Effective stress (${\sigma'_{zz}}$)",fontsize=f2)
    ax[1][1].set_title(r"Shear stress ($\sigma_{xz})$",fontsize=f2)

    
    ax[0][0].grid(True)
    ax[0][1].grid(True)
    ax[1][0].grid(True)
    ax[1][1].grid(True)

    #plt.legend()
    
    plt.colorbar(graph_omega)
    plt.colorbar(graph_vol_balance)
    plt.colorbar(graph_eff_stress)
    plt.colorbar(graph_shear_stress)

    fig.savefig("Other_Solutions_"+model+".png")
    print("fig saved")

    #plt.show()
    plt.close()


#3D plot solution for one variable 
def plot_solution_3d(coeff, dcoeff, x_lst, model, title, symb):
    fig = plt.figure(figsize=(17,10))
    graph = ax.scatter(x_lst[0], x_lst[1], coeff, c=coeff, s=0.01)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_zlabel(symb)
    ax.set_title(title)
    fig.savefig("3D_"+title+"_"+model+".png")
    print("3d fig saved")
    #plt.show()
    plt.close()
    

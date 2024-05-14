import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import fem_functions_2d
import matplotlib.patches as patches
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#plot the transient hydraulic load due to the water waves
def plot_load_Fzz(t=92.25):
    x = np.linspace(0,1,51)
    gamma_w=10**4
    load = fem_functions_2d.f_lab(t,x,gamma_w=gamma_w, T=9, H=3.5, Nc = 10, D=5.2)/gamma_w
    load_0 = fem_functions_2d.f_lab(t,x,gamma_w=gamma_w, T=9, H=3.5, Nc = 10, D=0)/gamma_w
    x_arrow = np.array([0, 0.5, 1])
    z_lst = np.cos(2.0*np.pi*x_arrow)
    c_lst = []
    d_lst = []

    
    for i in range(len(z_lst)):
        if (load[1]-load[0]) != 0:
            z = z_lst[i]
            print(z)
            if z > 0:
                c_lst += [0.8*z]
                d_lst += [0.8]
            else:
                c_lst += [0.8*z]
                d_lst += [0]
        else:
            c_lst += [0]
            d_lst += [0]
        
    fig, ax = plt.subplots(1,1,layout="constrained")
    ax.plot(x,load, color='black')
    ax.plot(x,load, '--', color='firebrick')
    #ax.plot(x,load_0, '--', color='firebrick')
    y = np.zeros(len(x))
    ax.plot(x, y, color='black')
    ax.plot(x,y, '--', color='firebrick')
    ax.fill_between(x,y,-2,color='moccasin')
    ax.fill_between(x,load,color='cornflowerblue')
    ax.fill_between(x,load,10,color='lightskyblue')
    ax.plot([0.1,0.106,0.112],[8.5,8.42,8.5], color='black')
    ax.plot([0.9,0.906,0.912],[8.75,8.67,8.75], color='black')
    ax.plot([0.6,0.606,0.612],[8,7.92,8], color='black')
    x=np.random.uniform(0.01,0.99,200)
    y=np.random.uniform(-1.99,-0.01,200)
    ax.scatter(x,y, color='darkgoldenrod', s=0.01)
    ax.arrow(0.15, 2+d_lst[0], 0, -c_lst[0], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[0])*10, color='firebrick', length_includes_head=True)
    ax.arrow(0.5, 2+d_lst[1], 0, -c_lst[1], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[1])*10, color='firebrick', length_includes_head=True)
    ax.arrow(0.85, 2+d_lst[2], 0, -c_lst[2], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[2])*10, color='firebrick', length_includes_head=True)

    ax.arrow(0.15,0.2+d_lst[0], 0, -c_lst[0], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[0])*10, color='firebrick', length_includes_head=True)
    ax.arrow(0.5, 0.2+d_lst[1], 0, -c_lst[1], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[1])*10, color='firebrick', length_includes_head=True)
    ax.arrow(0.85, 0.2+d_lst[2], 0, -c_lst[2], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[2])*10, color='firebrick', length_includes_head=True)
    ax.text(0.42,0.15, r'$F_{zz}(t)$', color='firebrick', fontsize=24)
    ax.text(0.915,9.0, 'Sky', color='black', fontsize=26)
    ax.text(0.915,0.5, 'Water', color='black', fontsize=26)
    ax.text(0.915,-1.75, 'Sand', color='black', fontsize=26)
    #ax.plot([0,1],[5.2,5.2],'--', color='blue')
    #ax.text(0.87,5.3, 'Equilibrium', color='blue', fontsize=26)
    ax.set_xlabel(r'$x$ [m]',fontsize=30)
    ax.set_ylabel(r'$z$ [m]',fontsize=30)
    ax.set_title(r'Water wave at $t = $'+ str(t) +' s', fontsize=35)
    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.show()
    fig.savefig('Wave_image_Fzz_t='+str(t)+'.png')
    plt.close()

#plot the dynamic pressure due to the water waves
def plot_load_pressure(t=92.25):
    x = np.linspace(0,1,51)
    gamma_w=10**4
    load = fem_functions_2d.f_lab(t,x,gamma_w=gamma_w, T=9, H=3.5, Nc = 10, D=5.2)/gamma_w
    load_0 = fem_functions_2d.f_lab(t,x,gamma_w=gamma_w, T=9, H=3.5, Nc = 10, D=0)/gamma_w
    x_arrow = np.array([0, 0.5, 1])
    z_lst = np.cos(2.0*np.pi*x_arrow)
    c_lst = []
    d_lst = []

    
    for i in range(len(z_lst)):
        if (load[1]-load[0]) != 0:
            z = z_lst[i]
            print(z)
            if z > 0:
                c_lst += [0.8*z]
                d_lst += [0.8]
            else:
                c_lst += [0.8*z]
                d_lst += [0]
        else:
            c_lst += [0]
            d_lst += [0]
        
    fig, ax = plt.subplots(1,1,layout="constrained")
    ax.plot(x,load, color='black')
    ax.plot(x,load, '--', color='firebrick')
    #ax.plot(x,load_0, '--', color='firebrick')
    y = np.zeros(len(x))
    ax.plot(x, y, color='black')
    ax.plot(x,y, '--', color='firebrick')
    ax.fill_between(x,y,-2,color='moccasin')
    ax.fill_between(x,load,color='cornflowerblue')
    ax.fill_between(x,load,10,color='lightskyblue')
    ax.plot([0.1,0.106,0.112],[8.5,8.42,8.5], color='black')
    ax.plot([0.9,0.906,0.912],[8.75,8.67,8.75], color='black')
    ax.plot([0.6,0.606,0.612],[8,7.92,8], color='black')
    x=np.random.uniform(0.01,0.99,200)
    y=np.random.uniform(-1.99,-0.01,200)
    ax.scatter(x,y, color='darkgoldenrod', s=0.01)
    print(d_lst, c_lst)
    ax.arrow(0.15, 1.6+d_lst[0], 0, -c_lst[0], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[0])*10, color='firebrick', length_includes_head=True)
    ax.arrow(0.5, 1.6+d_lst[1], 0, -c_lst[1], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[1])*10, color='firebrick', length_includes_head=True)
    ax.arrow(0.85, 1.6+d_lst[2], 0, -c_lst[2], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[2])*10, color='firebrick', length_includes_head=True)
    ax.text(0.435,0.88, 'Pressure', color='firebrick', fontsize=26)
    ax.text(0.915,9.0, 'Sky', color='black', fontsize=26)
    ax.text(0.915,0.5, 'Water', color='black', fontsize=26)
    ax.text(0.915,-1.75, 'Sand', color='black', fontsize=26)
    
    #ax.plot([0,1],[5.2,5.2],'--', color='blue')
    #ax.text(0.87,5.3, 'Equilibrium', color='blue', fontsize=26)
    ax.set_xlabel(r'$x$ [m]',fontsize=30)
    ax.set_ylabel(r'$z$ [m]',fontsize=30)
    #ax.set_title(r'Water wave', fontsize=35)
    ax.axis('off')
    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.show()
    fig.savefig('Wave_image_pressure_t='+str(t)+'.png')
    plt.close()

#make an animation of the dynamic pressure due to the water waves over time
def movie_load():
    fig, ax = plt.subplots(1,1,layout="constrained")
    # duration of the video
    duration = 9
    x = np.linspace(0,1,51)
    #t = np.linspace(0,92.25,9226)
    Nc = 10
    T = 9
    gamma_w=10**4
    t_end = (Nc+0.25)*T
    x2 = np.random.uniform(0.01,0.99,200)
    y2 = np.random.uniform(-1.99,-0.01,200)
    y3 = np.zeros(len(x))

    x_arrow = np.array([0, 0.5, 1])
    arrow = np.cos(2.0*np.pi*x_arrow)
            
    def make_frame(t):
        z_lst = np.sin(2.0*np.pi*t/T)*arrow
        c_lst = []
        d_lst = []
        for i in range(len(z_lst)):
            z = z_lst[i]
            if z > 0:
                c_lst += [0.8*z]
                d_lst += [0.8]
            else:
                c_lst += [0.8*z]
                d_lst += [0]
                
        ax.clear()
        load = fem_functions_2d.f_lab(t,x,gamma_w=gamma_w, T=T, H=3.5, Nc = Nc, D=5.2)/gamma_w
        #load_s = fem_functions_2d.f_lab(t,x,gamma_w=gamma_w, T=T, H=3.5, Nc = Nc, D=0)/gamma_w    
        ax.plot(x,load, color='black')
        ax.plot(x,load, '--', color='firebrick')
        #ax.plot(x,load_s, '--', color='firebrick')
        ax.plot(x, y3, color='black')
        ax.fill_between(x,y3,-2,color='moccasin')
        ax.fill_between(x,load,color='cornflowerblue')
        ax.fill_between(x,load,10,color='lightskyblue')

        ax.plot([0.1,0.106,0.112],[8.5,8.42,8.5], color='black')
        ax.plot([0.9,0.906,0.912],[8.75,8.67,8.75], color='black')
        ax.plot([0.6,0.606,0.612],[8,7.92,8], color='black')
        ax.scatter(x2,y2, color='darkgoldenrod', s=0.01)
        ax.plot([0,1],[5.2,5.2],'--', color='blue')
        ax.text(0.72,5.3, 'Equilibrium', color='blue', fontsize=16)

        ax.arrow(0.15, 1.15+d_lst[0], 0, -c_lst[0], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[0])*10, color='firebrick', length_includes_head=True)
        ax.arrow(0.5, 1.15+d_lst[1], 0, -c_lst[1], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[1])*10, color='firebrick', length_includes_head=True)
        ax.arrow(0.85, 1.15+d_lst[2], 0, -c_lst[2], head_width=0.015, head_length=0.3, linewidth=abs(c_lst[2])*10, color='firebrick', length_includes_head=True)
        #ax.text(0.25,2.35, 'Hydrodynamic load', color='firebrick', fontsize=16)
        ax.text(0.25,2.35, 'Pressure', color='firebrick', fontsize=16)
        #ax.text(0.005,7.1, r'$F_{zz}$'+f'({t})', color='firebrick', fontsize=20)

        ax.set_xlabel(r'$x$ [m]',fontsize=30)
        ax.set_ylabel(r'$z$ [m]',fontsize=30)
        ax.set_title(r'Water wave at $t = $'+str(round(t,2))+'s  ', fontsize=35)
        plt.xticks(fontsize = 26)
        plt.yticks(fontsize = 26)
        # returning numpy image
        return mplfig_to_npimage(fig)
    

    # creating animation
    animation = VideoClip(make_frame, duration = duration)

    # displaying animation with auto play and looping
    #animation.ipython_display(fps = 20, loop = True, autoplay = True)
    animation.write_gif('load_animation.gif', fps=10)
    t_stop = ['00:00:00', '00:00:03', '00:00:06', '00:00:09']
    t = 0
    for i in t_stop:
        t+=1
        animation.save_frame("arrows_load_frame_t="+str(t)+'.png', t=i)

#plot the shear stress due to the water waves
def plot_shear_Fxz():
    fig, ax = plt.subplots(1,2,layout="constrained")
    d = 0.12 #mean grain size [mm]
    r = d/2 #radius [mm]
    num_x = 3 #amount of soil particles in x-direction
    num_y = 3 #amount of soil particles in z-direction
    xlim = [0,num_x*d]
    ylim = [-num_y*d,0]
    xlim2 = [0-r,(num_x+2)*d]
    ylim2 = [-num_y*d-r,0+r]
    mid_x = (xlim[1]+xlim[0])/2
    mid_y = (ylim[1]+ylim[0])/2
    y0 = ylim[0] + r
    y1 = ylim[0] + r
    i=1
    while y0 >= ylim[0] and y0 <= ylim[1] and i<=num_y:
        x0 = xlim[0] + r
        i += 1
        j = 1
        while x0 >= xlim[0] and x0 <= xlim[1] and j<=num_x:
            circle0 = plt.Circle((x0,y0),radius=r, facecolor='moccasin', edgecolor='black')
            ax[0].add_patch(circle0)
            x0 += d
            j+=1
        y0 += d 

    i = 1
    while y1 >= ylim2[0] and y1 <= ylim2[1] and i<=num_y:
        x1 = xlim[0] + r
        i += 1
        j = 1
        while x1 >= xlim2[0] and x1 <= xlim2[1] and j<=2*num_x-1:
            circle1 = plt.Circle((x1,y1),radius=r, facecolor='moccasin', edgecolor='black')
            ax[1].add_patch(circle1)
            x1 += d
            j+=1 
        y1 += d 
    square0 = plt.Rectangle((mid_x-d-r,mid_y-d-r),num_x*d,num_y*d,facecolor='none', edgecolor='black')
    print('mid = ', mid_x, mid_y)
    x0 = mid_x-d-r
    x1 = x0 + num_x*d
    x2 = x1 + 2*d
    x3 = x2 - (x1-x0)
    y0 = ylim[0] 
    y1 = ylim[0]
    y2 = ylim[1]
    y3 = ylim[1]
    p = list(zip([x0,x1,x2,x3],[y0, y1, y2, y3]))
    p1 = list(zip([x0-0.001, x3, x0-0.001], [y0, y3+0.001, y3+0.001]))
    p2 = list(zip([x1, x2+0.001, x2+0.001], [y1-0.001, y1-0.001, y2]))
   
    parallel1 = patches.Polygon(p,fill=False, color='black')
    triangle1 = plt.Polygon(p1, fill=True, color='white')
    triangle2 = plt.Polygon(p2, fill=True, color='white') 
    ax[0].add_patch(square0)
    ax[1].add_patch(triangle1)
    ax[1].add_patch(triangle2)
    ax[1].add_patch(parallel1)
    ax[0].arrow(xlim[0]+r, ylim[1]+r/2, 0.23, 0, head_width=0.025, head_length=0.03, linewidth=8, color='firebrick', length_includes_head=True) #up
    ax[0].arrow(xlim[1]-r, ylim[0]-r/2, -0.23, 0, head_width=0.025, head_length=0.03, linewidth=8, color='firebrick', length_includes_head=True) #down
    ax[0].arrow(xlim[0]-r/2, ylim[1]-r , 0, -0.23, head_width=0.025, head_length=0.03, linewidth=8, color='firebrick', length_includes_head=True) #left
    ax[0].arrow(xlim[1]+r/2, ylim[0]+r , 0, 0.23, head_width=0.025, head_length=0.03, linewidth=8, color='firebrick', length_includes_head=True) #right
    ax[0].set_xlim(xlim2)
    ax[1].set_xlim(xlim2)
    ax[0].set_ylim(ylim2)
    ax[1].set_ylim(ylim2)
    ax[0].text(xlim[0]+3*r/2,ylim[1]+3*r/4, 'Shear stress', color='firebrick', fontsize=16)
    ax[0].text(r/32,-9*r/8, 'Soil particle', color='black', fontsize=16)
    ax[1].text(x3+r/16,-9*r/8, 'Soil particle', color='black', fontsize=16)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    plt.show()
    fig.savefig('Soil_particle_image_Fxz.png')
    plt.close()

#plot the normal stress due to the water waves
def plot_normal():
    fig, ax = plt.subplots(1,2,layout="constrained")
    d = 0.12 #mean grain size [mm]
    r = d/2 #radius [mm]
    num_x = 3 #amount of soil particles in x-direction
    num_y = 4 #amount of soil particles in z-direction
    xlim = [0,num_x*d]
    ylim = [-num_y*d,0]
    xlim1 = [-r,num_x*d+r]
    ylim1 = [-num_y*d+r,-r]
    xlim2 = [-d,(num_x+1.5)*d]
    ylim2 = [-(num_y+1.5)*d,d]
    mid_x = (xlim[1]+xlim[0])/2
    mid_y = (ylim[1]+ylim[0])/2
    
    y0 = ylim[0]+r
    i=1
    k=1
    while y0 >= ylim[0] and y0 <= ylim[1] and i<=num_y:
        x0 = xlim[0] + r
        i += 1
        j = 1
        while x0 >= xlim[0] and x0 <= xlim[1] and j<=num_x:
            circle0 = plt.Circle((x0,y0),radius=r, facecolor='moccasin', edgecolor='black')
            ax[0].add_patch(circle0)
            x0 += d
            j+=1
        y0 += d
        
    y1 = ylim1[0]
    i=1    
    while y1 >= ylim1[0] and y1 <= ylim1[1] and k<=num_y*num_x:
        x1 = xlim1[0] + d
        i += 1
        j = 1
        while x1 >= xlim1[0] and x1 <= xlim1[1]:
            circle1 = plt.Circle((x1,y1),radius=r, facecolor='moccasin', edgecolor='black')
            ax[1].add_patch(circle1)
            x1 += d
            j+=1
            k+=1
        y1 += d
    square0 = plt.Rectangle((mid_x-d-r,mid_y-2*d),num_x*d,num_y*d,facecolor='none', edgecolor='black')
    square1 = plt.Rectangle((mid_x-d-r,mid_y-2*d),(num_x+1)*d,(num_y-1)*d,facecolor='none', edgecolor='black')
    
    ax[0].add_patch(square0)
    ax[1].add_patch(square1)
    ax[0].arrow(mid_x, ylim[1]+0.07, 0, -0.05, head_width=0.02, head_length=0.015, linewidth=8, color='firebrick', length_includes_head=True) #up
    ax[0].arrow(mid_x, ylim[0]-0.07, 0, 0.05, head_width=0.02, head_length=0.015, linewidth=8, color='firebrick', length_includes_head=True) #down
    ax[0].arrow(xlim[0]+r-0.07, mid_y, -0.05, 0, head_width=0.02, head_length=0.015, linewidth=8, color='firebrick', length_includes_head=True) #left
    ax[0].arrow(xlim[1]-r+0.07, mid_y, 0.05, 0, head_width=0.02, head_length=0.015, linewidth=8, color='firebrick', length_includes_head=True) #right
    ax[0].set_xlim(xlim2)
    ax[1].set_xlim(xlim2)
    ax[0].set_ylim(ylim2)
    ax[1].set_ylim(ylim2)
    ax[0].text(xlim[0]+r/8,ylim[1]+3*r/4, 'Normal stress', color='firebrick', fontsize=16)
    ax[0].text(r/64,-9*r/8, 'Soil particle', color='black', fontsize=14)
    ax[1].text(r/64,-26*r/16-6*r/4, 'Soil particle', color='black', fontsize=14)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    plt.show()
    fig.savefig('Soil_particle_image_normal_stress.png')
    plt.close()

#plot the density of the soil
def plot_density():
    fig, ax = plt.subplots(1,1,layout="constrained")
    fig2, ax2 = plt.subplots(1,1,layout="constrained")
    d = 0.12 #mean grain size [mm]
    r = d/2 #radius [mm]
    num_x = 3 #amount of soil particles in x-direction
    num_y = 3 #amount of soil particles in z-direction
    xlim = [0,num_x*d]
    ylim = [-num_y*d,0]
    xlim2 = [-d,(num_x+1.5)*d]
    ylim2 = [-(num_y+1.5)*d,d]
    mid_x = (xlim[1]+xlim[0])/2
    mid_y = (ylim[1]+ylim[0])/2
    y0 = ylim[0] + r
    y1 = ylim[0] + r
    i=1
    k=1
    while y0 >= ylim[0] and y0 <= ylim[1] and i<=num_y:
        x0 = xlim[0] + r
        x1 = xlim[0] + (k/2)*r
        k *= -1
        i += 1
        j = 1
        while x0 >= xlim[0] and x0 <= xlim[1] and j<=num_x:
            circle0 = plt.Circle((x0,y0),radius=r, facecolor='moccasin', edgecolor='black')
            circle1 = plt.Circle((x1,y1),radius=r, facecolor='moccasin', edgecolor='black')
            ax.add_patch(circle0)
            ax2.add_patch(circle1)
            x0 += d
            x1 += d
            j+=1
        y0 += d 
        y1 += d - d/8
    ax.set_xlim(xlim2)
    ax2.set_xlim(xlim2)
    ax.set_ylim(ylim2)
    ax2.set_ylim(ylim2)
    ax.text(r/32,-9*r/8, 'Soil particle', color='black', fontsize=16)
    ax2.text(-15*r/32,-25*r/16, 'Soil particle', color='black', fontsize=16)
    ax.axis('off')
    ax2.axis('off')
    ax.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    
    plt.show()
    fig.savefig('Soil_particle_image_Loose.png')
    fig2.savefig('Soil_particle_image_Dense.png')
    plt.close()

    
if __name__ == "__main__":
    plot_load_Fzz(2.25)
    plot_load_pressure(2.25)
    #movie_load()

    plot_normal()
    plot_shear_Fxz()
    plot_density()

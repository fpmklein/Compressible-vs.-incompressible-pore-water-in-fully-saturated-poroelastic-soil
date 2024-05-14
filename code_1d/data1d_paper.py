import matplotlib.pyplot as plt
import numpy as np
import fem_functions_1d
from collections import defaultdict

def time(hour, minute, second):
    t = hour*3600 + minute*60 + second
    return t

def time_xticks(hour, minute, second):
    t = hour*3600 + minute*60 + second
    str_hour = str(hour)
    str_minute = str(minute)
    str_sec = str(second)
    if len(str_hour) == 1:
        str_hour = str(0) + str_hour
    if len(str_minute) == 1:
        str_minute = str(0) + str_minute
    if len(str_sec) == 1:
        str_sec = str(0) + str_sec
    label = str_hour + ":" + str_minute + ":" + str_sec
    lst = np.array([t, label])
    return lst

def data(ex = ''):
    xticks = np.array([])
    if ex == 'boat_a':
        #26-10-21 09:13:30
        #air pressure = 10212
        #P [m], t [s]
        a = np.array([  3.875, 3.875, 3.93, 3.91, 3.875, 3.54,
                        3.675, 3.695, 3.67, 3.6, 3.82,
                        3.725, 3.905, 3.88, 3.88, 3.88]) #P1
        
        d = np.array([  3.78, 3.78, 3.79, 3.793, 3.783, 3.713,
                        3.675, 3.66, 3.635, 3.625, 3.685,
                        3.68, 3.755, 3.775, 3.798, 3.798]) #+ (3.88-3.765) #P2
        
        c = np.array([]) #P3
        
        b = np.array([  3.695, 3.695, 3.7, 3.707, 3.697, 3.627,
                        3.59, 3.565, 3.54, 3.535, 3.585,
                        3.585, 3.678, 3.688, 3.708, 3.708]) #+ (3.88-3.69)
        
        z = np.array([  time(9,12,30), time(9,13,3), time(9,13,5), time(9,13,7), time(9,13,8), time(9,13,9),
                        time(9,13,12), time(9,13,16), time(9,13,23), time(9,13,25), time(9,13,28),
                        time(9,13,30), time(9,13,34), time(9,13,37), time(9,13,47), time(9,14,30)])
        
        xticks = np.array([ time_xticks(9,12,23), time_xticks(9,12,58), time_xticks(9,13,32),
                            time_xticks(9,14,7), time_xticks(9,14,42)])
        
    elif ex == 'boat_b':
        #27-10-21 17:49:28
        #air pressure = 10212
        #P [m], t [s]
        a = np.array([  3.88, 3.88, 3.915, 3.865, 3.79, 3.78,
                        3.79, 3.79, 3.88, 3.65, 3.8,
                        3.9, 3.88, 3.88, 3.88]) #P1
        
        d = np.array([  3.765, 3.765, 3.78, 3.745, 3.725, 3.71,
                        3.7, 3.695, 3.72, 3.675, 3.665,
                        3.72, 3.74, 3.767, 3.77]) #+ (3.88-3.765) #P2
        
        c = np.array([]) #P3
        
        b = np.array([  3.69, 3.69, 3.71, 3.66, 3.64, 3.625,
                        3.615, 3.61, 3.645, 3.595, 3.585,
                        3.645, 3.665, 3.695, 3.698]) #+ (3.88-3.69)
        
        z = np.array([  time(17,48,29), time(17,49,6), time(17,49,13), time(17,49,15), time(17,49,18), time(17,49,20),
                        time(17,49,21), time(17,49,23), time(17,49,25), time(17,49,26), time(17,49,28),
                        time(17,49,29), time(17,49,32), time(17,49,38), time(17,50,29)])
        xticks = np.array([ time_xticks(17,47,54), time_xticks(17,48,29), time_xticks(17,49,3),
                            time_xticks(17,49,38), time_xticks(17,50,12), time_xticks(17,50,47)])
        
    elif ex == 'boat_c': #rws
        #28-10-21 15:41:56
        #P [m], t [s]
        a = np.array([  3.74, 3.74, 3.73, 3.73, 3.72,
                        3.725, 3.725, 3.73, 3.735, 3.755, 3.74,
                        3.73, 3.575, 3.4, 3.465, 3.464, 3.466, 3.465, 3.44,
                        3.44, 3.43, 3.43, 3.74, 3.755,
                        3.735, 3.745, 3.745, 3.735, 3.735,
                        3.73, 3.735, 3.73, 3.725, 3.73]) #P1
        
        d = np.array([  3.74, 3.74, 3.73, 3.73, 3.72,
                        3.725, 3.725, 3.73, 3.734, 3.740, 3.74,
                        3.732, 3.645, 3.565, 3.55, 3.535, 3.515, 3.5, 3.495,
                        3.49, 3.489, 3.49, 3.635, 3.71,
                        3.745, 3.75, 3.76, 3.755, 3.76,
                        3.755, 3.76, 3.755, 3.75, 3.755]) - 0.07 #P2 #steady state
        
        c = np.array([]) #P3
        
        b = np.array([  3.74, 3.74, 3.73, 3.73, 3.72,
                        3.725, 3.725, 3.73, 3.735, 3.737, 3.745,
                        3.734, 3.72, 3.57, 3.55, 3.51, 3.495, 3.485, 3.48,
                        3.4775, 3.475, 3.48, 3.635, 3.715,
                        3.75, 3.755, 3.765, 3.76, 3.765,
                        3.76, 3.765, 3.76, 3.755, 3.76]) - 0.145 #P4 #steady state
        
        z = np.array([  time(15,40,54), time(15,40,59), time(15,41,5), time(15,41,26), time(15,41,29),
                        time(15,41,32), time(15,41,35), time(15,41,39), time(15,41,51), time(15,41,54), time(15,41,56), 
                        time(15,41,58), time(15,41,59), time(15,42,0), time(15,42,1), time(15,42,3), time(15,42,8), time(15,42,12), time(15,42,14),
                        time(15,42,16), time(15,42,17), time(15,42,19), time(15,42,23), time(15,42,25),
                        time(15,42,29), time(15,42,31), time(15,42,35), time(15,42,37), time(15,42,39),
                        time(15,42,41), time(15,42,46), time(15,42,48), time(15,42,52), time(15,42,54)])
        xticks = np.array([ time_xticks(15,40,36), time_xticks(15,41,11), time_xticks(15,41,46),
                            time_xticks(15,42,20), time_xticks(15,42,55)])    
    elif ex == 'boat_d':
        #01-11-21 09:01:03
        #air pressure = 10212
        #P [m], t [s]
        a = np.array([  3.645, 3.65, 3.66, 3.655, 3.68, 3.625,
                        3.56, 3.505, 3.675, 3.661, 3.662,
                        3.665, 3.665, 3.668]) #P1
        
        d = np.array([  3.59, 3.595, 3.61, 3.603, 3.61, 3.58,
                        3.54, 3.5, 3.59, 3.605, 3.61,
                        3.612, 3.612, 3.6121]) #+ (3.88-3.765) #P2
        
        c = np.array([]) #P3
        
        b = np.array([  3.505, 3.51, 3.53, 3.52, 3.53, 3.505,
                        3.44, 3.3925, 3.515, 3.53, 3.528,
                        3.5275, 3.5276, 3.528]) #+ (3.88-3.69)
        
        z = np.array([  time(9,0,3), time(9,0,17), time(9,0,27), time(9,0,36), time(9,0,51), time(9,0,55),
                        time(9,1,0), time(9,1,10), time(9,1,16), time(9,1,20), time(9,1,36),
                        time(9,1,42), time(9,1,57), time(9,2,2)])

        xticks = np.array([ time_xticks(8,59,43), time_xticks(9,0,17), time_xticks(9,0,52),
                            time_xticks(9,1,26), time_xticks(9,2,1), time_xticks(9,2,36)]) 
     
    elif ex == 'density': #lab
        #P_norm [Pa/Pa], z_norm [m/m]
        #when varying the density
        aDense_x = [0.54, 0.58, 0.67, 0.73, 0.759, 0.82, 0.88, 0.95, 0.99, 1.0]
        aLoose_x = [0.84, 0.855, 0.875, 0.88, 0.885, 0.895, 0.94, 0.985, 0.99, 1.0]
        a = np.array([aDense_x, aLoose_x])

        bDense_x = [0.405, 0.42, 0.44, 0.48, 0.49, 0.55, 0.575, 0.67, 0.715, 1.0]
        bLoose_x = [0.48, 0.51, 0.57, 0.62, 0.645, 0.7, 0.81, 0.93, 0.98, 1.0]
        b = np.array([bDense_x, bLoose_x])
        
        cDense_x = [0.465, 0.485, 0.565, 0.635, 0.675, 0.735, 0.79, 0.87, 0.905, 1.1]
        cLoose_x = [0.52, 0.55, 0.62, 0.685, 0.73, 0.785, 0.85, 0.96, 0.99, 1.0]
        c = np.array([cDense_x, cLoose_x])

        dDense_x = [0.4, 0.43, 0.465, 0.51, 0.52, 0.575, 0.615, 0.72, 0.77, 1.0]
        dLoose_x = [0.5, 0.515, 0.535, 0.56, 0.57, 0.61, 0.625, 0.7, 0.735, 1.0]
        d = np.array([dDense_x, dLoose_x])

        z = np.array([-0.615, -0.5, -0.385, -0.29, -0.27, -0.18, -0.15, -0.07, -0.035, 0.0])
       #Dense, loose sand
        
    elif ex == 'saturation': #lab
        #P_norm [Pa/Pa], z_norm [m/m]
        #when varying the saturation degree
        aLowSat_x = [0.42, 0.47, 0.55, 0.63, 0.66, 0.74, 0.77, 0.89, 0.93, 1.0]
        aHighSat_x = [0.84, 0.85, 0.86, 0.87, 0.88, 0.905, 0.945, 0.98, 0.95, 1.0]
        a = np.array([aLowSat_x, aHighSat_x])

        bLowSat_x = [0.18, 0.2, 0.26, 0.28, 0.3, 0.47, 0.55, 0.65, 0.7, 1.0]
        bHighSat_x = [0.5, 0.525, 0.565, 0.62, 0.66, 0.77, 0.8, 0.92, 0.97, 1.0]
        b = np.array([bLowSat_x, bHighSat_x])

        cLowSat_x = [0.32, 0.38, 0.48, 0.57, 0.605, 0.685, 0.735, 0.85, 0.89, 1.0]
        cHighSat_x = [0.51, 0.55, 0.61, 0.635, 0.695, 0.75, 0.79, 0.85, 0.96, 1.0]
        c = np.array([cLowSat_x, cHighSat_x])

        dLowSat_x = [0.14, 0.175, 0.21, 0.35, 0.375, 0.435, 0.505, 0.605, 0.665, 1.0]
        dHighSat_x = [0.49, 0.51, 0.52, 0.56, 0.57, 0.605, 0.62, 0.71, 0.73, 1.0]
        d = np.array([dLowSat_x, dHighSat_x])

        z = np.array([-0.615, -0.5, -0.385, -0.29, -0.27, -0.18, -0.15, -0.07, -0.035, 0.0])    
        #High, low saturation
    else: #lab
        #P_norm [Pa/Pa], z_norm [m/m]
        a = np.array([1.0, 0.95, 0.83, 0.8, 0.73, 0.7, 0.6, 0.55, 0.55, 0.5])
        b = np.array([0])
        c = np.array([0])
        d = np.array([0])
        z = -np.array([0.0, 0.075, 0.15, 0.18, 0.25, 0.3, 0.4, 0.5, 0.6, 0.85])
    return z, a, b, c, d, xticks

    
if __name__ == "__main__":
    t,P1,P2,P3,P4,xticks = data('boat_d')
    t_new, P1_new = fem_functions_1d.f_boat(0.2, t, P1)
    
    i = 0
    d2 = P1[i] - P2[i]
    d4 = P1[i] - P4[i]
    plt.plot(t,P1,'blue', label = 'P1')#, label = 'data')
    #plt.plot(t_new,P1_new,'purple', label = 'new')
    plt.plot(t,P2,'pink', label = 'steady P2')
    plt.plot(t,P4,'red', label = 'steady P4') #due to steady state, here P4<P2 at bottom instead of P4>P2
    plt.plot(t,P2+d2,'orange', label = 'P2')
    plt.plot(t,P4+d4,'yellow', label = 'P4')
    plt.legend()
    plt.grid('True')
    plt.show()

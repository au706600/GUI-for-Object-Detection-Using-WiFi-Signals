import numpy as np
import math

def array_response_ULA(theta, N): 

    m = np.arange(N)

    y = np.exp(1j * np.pi * m * np.sin(theta))

    return y

# -----------------------------


import math

def angles(TX_pos, RX_pos):
    # Coordinate in the 3D space
    x=RX_pos[0]-TX_pos[0]
    y=RX_pos[1]-TX_pos[1]
    z=RX_pos[2]-TX_pos[2]
    
    d = np.sqrt(x**2 + y**2 + z**2)
    
    d2D = np.sqrt(x**2 + y**2)
    
    theta = np.arctan2(z,d2D)
    
    phi = np.arctan2(y,x)

    return d, d2D, theta, phi


# --------------------------------------------

import numpy as np

def generate_channel(TX_pos,RX_pos,object_pos,Nt, Nr,f):

    c = 3*10**8

    [d_r,_,_,phi_r] = angles(TX_pos,RX_pos)

    attenuation_LOS= 1/(d_r)*1/(4*math.pi*f/c)

    at = array_response_ULA(phi_r,Nt)

    ar = array_response_ULA(-phi_r,Nr)

    #H_LOS=attenuation_LOS*np.transpose(np.conj(at)) @ ar

    #H_LOS = attenuation_LOS * np.transpose(np.conj(at)) * ar

    H_LOS = attenuation_LOS * np.outer(np.conj(at), ar)

    reflection_coefficient = 0.8

    [d_obj_rx,_,_,phi_obj_rx]=angles(object_pos,RX_pos)

    attenuation_obj_rx = 1/(d_obj_rx) * 1/(4*np.pi*f/c)

    at_obj_rx = array_response_ULA(phi_obj_rx, 1)

    ar_obj_rx = array_response_ULA(-phi_obj_rx, Nr)

    #H_obj_rx = attenuation_obj_rx * np.transpose(np.conj(at_obj_rx)) @ ar_obj_rx

    #H_obj_rx = attenuation_obj_rx * np.transpose(np.conj(at_obj_rx)) * ar_obj_rx

    H_obj_rx = attenuation_obj_rx * np.outer(np.conj(at_obj_rx), ar_obj_rx)

    H_3hops = reflection_coefficient * H_LOS @ np.transpose(np.conj(H_obj_rx)) @ H_obj_rx @ np.transpose(np.conj(H_LOS))
    
    #H = H_LOS * np.transpose(H_LOS)

    #H = np.dot(H_LOS, np.transpose(H_LOS))

    H = np.outer(H_LOS, np.conj(H_LOS))
    
    H_w_obj = H_3hops

    return H, H_w_obj, H_3hops

# ---------------------------------------------


def generate_channel_RIS_weights(TX_pos, RX_pos, object_pos, Nt, NR, f, weights):

    global w 
    c = 3*10**8
    [d_r,_,_,phi_r] = angles(TX_pos, RX_pos)

    attenuation_LOS = 1/(d_r)*1/(4*np.pi*f/c)

    at = array_response_ULA(phi_r, Nt)
    ar = array_response_ULA(-phi_r, NR)


    #H_LOS = attenuation_LOS*np.transpose(np.conj(at)) @ ar

    #H_LOS = attenuation_LOS * np.transpose(np.conj(at)) * ar

    H_LOS = attenuation_LOS * np.outer(np.conj(at), ar)

    reflection_coefficient = 0.8


    [d_obj_rx,_,_,phi_obj_rx] = angles(object_pos,RX_pos)


    attenuation_obj_rx = 1/(d_obj_rx) * 1/(4*np.pi*f/c)


    at_obj_rx = array_response_ULA(phi_obj_rx, 1)


    ar_obj_rx = array_response_ULA(-phi_obj_rx, NR)

    #H_obj_rx = attenuation_obj_rx * np.transpose(np.conj(at_obj_rx)) * ar_obj_rx

    H_obj_rx = attenuation_obj_rx * np.outer(np.conj(at_obj_rx), ar_obj_rx)


    H_3hops = reflection_coefficient * H_LOS @ weights @ np.transpose(H_obj_rx) @ H_obj_rx @ weights @ np.transpose(np.conj(H_LOS));

    
    #gain1 = np.conj(at).T * ar * weights * np.transpose((np.transpose(np.conj(at_obj_rx)) * ar_obj_rx))

    gain1 = np.transpose(np.conj(at)) @ ar @ weights @ (at_obj_rx @ np.conj(ar_obj_rx));

    
    #gain0 = np.transpose(np.conj(at)) @ ar @ np.transpose(np.transpose(np.conj(at_obj_rx)) * ar_obj_rx) @ np.transpose(np.conj(at_obj_rx)) @ ar_obj_rx @ np.transpose(np.transpose(np.conj(at) * ar))

    gain0 = np.transpose(np.conj(at)) @ ar @ (at_obj_rx @ np.conj(ar_obj_rx)) @ at_obj_rx @ ar_obj_rx @ np.transpose(np.conj(at@ar));
    
    gain = math.sqrt(np.sum(abs(gain1)**2)/np.sum(abs(gain0)**2));

    
    H = H_3hops;
    
    H_w_obj = H_3hops;

    return H,H_w_obj,H_3hops,gain


# --------------------------------------

def Step_Search(lb, ub, f):
    nr = 5
    grid_size = 10
    
    G = np.linspace(lb, ub, grid_size)
    
    x_min = G[0]
    
    f_min = f(G[0])
    
    for r in range(nr):
        for i in range(len(G)):
            fvalue = f(G[i])
            if fvalue <= f_min:
                f_min = fvalue
                x_min = G[i]

        grid_size = grid_size*2
        G = np.linspace(lb , x_min, grid_size)
            
    initial_points = [x_min] 
    
    return initial_points

#import numpy as np

#def Step_Search(lb, ub, f):
#    nr = 5
#    grid_size = 10
    
#    G = np.linspace(lb, ub, grid_size)
    
#    x_min = G[0]
    
#    f_min = f(G[0])
    
#    length_of_G = len(G)
    
#    for r in range(nr):
#        for i in range(length_of_G):
#            fvalue = f(G[i])
#            if fvalue <= f_min:
#                f_min = fvalue
#                x_min = G[i]

#        grid_size = grid_size*2
#        G = np.linspace(lb , x_min, grid_size)
            
#    initial_point = x_min
    
#    return initial_point

# ------------------------------------------

from scipy.optimize import minimize
        
def ls_function_dist(pos, esti_theta,TX_pos, RIS_pos,X, y3, M, N,tau, f,risset):
    global w
    distance = pos
    theta = esti_theta[0]
    deltaX = distance * np.cos(np.deg2rad(theta))
    deltaY = distance * np.sin(np.deg2rad(theta))
    obj_pos = [RIS_pos[0] + deltaX, RIS_pos[1] + deltaY, 0] 

    if risset == 0:
        [_,_,H_3] = generate_channel(TX_pos, RIS_pos, obj_pos, M, N, f)
    
    else: 
        [_,_,H_3] = generate_channel_RIS_weights(TX_pos, RIS_pos, obj_pos, M, N, f, np.diag(w))
    
    s_3 = H_3@X
    #s_3 = H_3*X
    err = np.linalg.norm(s_3 - y3, 2)   

    return err

def ls_function(pos, TX_pos, RIS_pos, X, y3, M, N, tau, f, risset): 
    global w
    theta = pos
    distance = 3
    deltaX = distance * np.cos(np.deg2rad(theta))
    deltaY = distance * np.sin(np.deg2rad(theta))
    obj_pos = [RIS_pos[0] + deltaX, RIS_pos[1] + deltaY, 0]

    if risset == 0: 
        [_,_,H_3] = generate_channel(TX_pos, RIS_pos, obj_pos, M, N, f)
    
    else: 
        [_,_,H_3] = generate_channel_RIS_weights(TX_pos, RIS_pos, obj_pos, M, N, f, np.diag(w))
    
    s_3 = H_3 @ X
    #s_3 = H_3*X

    err = np.linalg.norm(s_3 - y3, 2)

    return err


# Taking snrdb, ietration and risset as arguments
# snrdb: SNR in dB
# ietration: number of iterations
# risset: whether RIS is used or not

# We have to use these in run_mse_esti(snrdb, ietration, risset): 

# Angle is given in radians, which is converted to degrees
# theta_real = np.degrees(np.arctan2(y2 - RIS_ypos, x2 - RIS_xpos))

# distance can be calculated by euclidean distance: 
# distance = np.sqrt((x - RIS_xpos)**2+(y - RIS_ypos)**2)

def run_mse_esti(distance, theta_real, snrdb, ietration, risset, coordinates = None):

    global w 

    esti_theta = np.zeros((ietration, 1))

    esti_d = np.zeros((ietration, 1))

    f = 2.4*10**9

    M = 1

    N = 64

    tau = 1 
    
    TX_pos = np.array([3, 0, 0])

    RIS_pos = np.array([0, 3, 0])


    if coordinates is not None:
        for x, y in coordinates:
            distance = np.sqrt((x - RIS_pos[0])**2+(y - RIS_pos[1])**2)

            theta_real = np.degrees(np.arctan2(y - RIS_pos[1], x - RIS_pos[0]))


    #distance = 3

    #theta_real = 10

    deltaX_r = distance * np.cos(np.deg2rad(theta_real))

    deltaY_r = distance * np.sin(np.deg2rad(theta_real))

    obj_pos = [RIS_pos[0] + deltaX_r, RIS_pos[1] + deltaY_r, 0]

    
    X = np.ones((M, tau))/np.sqrt(M)

    snr_deci = np.power(10, snrdb / 10)

    npower = 1

    Tx_p = np.power(10, 195 / 10)

    spower = npower * snr_deci * Tx_p

    samp = np.sqrt(spower)

    X = samp * X


# s3 = H_3h * x;               % propagate through channel s (M \times tau)  
    
    if risset == 0: 
        [_,_,H_3h] = generate_channel(TX_pos,RIS_pos,obj_pos,M, N,f)
    else:
        [_,_,H_3h] = generate_channel_RIS_weights(TX_pos,RIS_pos,obj_pos,M, N,f,np.diag(w))


    s3 = H_3h @ X
    #s3 = H_3h*X
        
    
    for i in range(ietration):
        n1 = (np.random.randn(M,tau)+1j*np.random.randn(M,tau))/np.sqrt(2)
 
        # real_snr = 20 * np.log10(np.linalg.norm(s3) / np.linalg.norm(n1))

        y3 = s3 + n1

        objFunc = lambda pos: ls_function(pos, TX_pos, RIS_pos, X, y3, M, N, tau, f, risset)


        lb = 0

        ub = 90
        
        initialPoints = Step_Search(lb, ub, objFunc)    

        
        results_theta = [minimize(objFunc, x0=point, bounds=[(lb, ub)]) for point in initialPoints]
        best_result_theta = min(results_theta, key=lambda x: x.fun)
        esti_theta[i] = best_result_theta.x

        
        objFunc_d = lambda dist: ls_function_dist(dist, esti_theta[i], TX_pos, RIS_pos, X, y3, M, N, tau, f, risset)

        lb_d = 1

        ub_d = 5 
        
        initialPoints_d = [(point,) for point in [0.5, 1.5, 2.5, 3.5, 4.5]]

        results_d = [minimize(objFunc_d, x0=point, bounds=[(lb_d, ub_d)]) for point in initialPoints_d]

        best_result_d = min(results_d, key=lambda x: x.fun)
        esti_d[i] = best_result_d.x
    
    return np.round(esti_theta, 3), np.round(esti_d,3)

# Function calls 
#esti_theta, esti_d = run_mse_esti(3, 10, 100, 10, 0)

#print("Estimated theta: ", esti_theta)

#print("Estimated distance: ", esti_d)

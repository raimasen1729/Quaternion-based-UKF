import numpy as np
from scipy import io
from quaternion import Quaternion
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
    #vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    # your code goes here
    #data = vicon.get('rots')
    np.random.seed((42))
    #samples = data.shape[2]
    #accel_true = vicon['rots'][0:3,:]
    #gyro_true = vicon['rots'][3:6,:]
    #T_true = np.shape(vicon['ts'])[1]

    #print(accel.shape)
    #print(accel)
    #print(gyro.shape)
    #print(gyro)
    #print(T)

    #print(accel_true.shape)
    #print(accel_true)
    #print(gyro_true.shape)
    #print(gyro_true)
    #print(T_true)


    accel_beta = [511, 501, 503]
    accel_alpha = 33.86 
    gyro_beta = [369.5, 371.5, 377]             
    gyro_alpha = 193.55

    accel_value = []
    gyro_value = []

    for i in range(T-1):
        acc = (accel[:,i]-accel_beta)*(3300/(1023*accel_alpha))
        accel_value.append(acc)
        gyr = (gyro[:,i]-gyro_beta)*(3300/(1023*gyro_alpha))
        gyro_value.append(gyr)
       
    ###### FINAL ACCELEROMETER AND GYROSCOPE VALUES IN PHYSICAL UNITS ######
    accel_value = np.transpose(np.array(accel_value))       #gravity #3x5561            #relates on quaternion componenent
    gyro_value = np.transpose(np.array(gyro_value))         #angular velocity #3x5561   #relates on omega component

    accel_value[0] = - accel_value[0]           #making changes according to convention
    accel_value[1] = - accel_value[1]

    gz = gyro_value[0]
    gx = gyro_value[1]
    gy = gyro_value[2]

    gyro_value = np.vstack((gx, gy, gz))

    #print(accel_value)
    #print(accel_value.shape)
    #print(gyro_value)
    #print(gyro_value.shape)

    accel_pitch = np.arctan2(accel_value[0], accel_value[2])
    accel_roll = np.arctan2(accel_value[1], accel_value[2])

    #print(accel_roll.shape)

    obs = np.vstack((accel_value,gyro_value))

    #print(obs.shape)
    '''
    ###### VICON DATA ARRANGEMENT ######
    true_roll = []
    true_pitch = []
    true_yaw = []

    for i in range(T-1):
        r = data[:, :, i]
        q = Quaternion()
        q.from_rotm(r)
        gt_roll, gt_pitch, gt_yaw = q.euler_angles()
        true_roll.append(gt_roll)
        true_pitch.append(gt_pitch)
        true_yaw.append(gt_yaw)

    ###### PLOTTING VICON DATA ######
    plt.subplot(2,1,1)
    plt.plot(np.arange(data.shape[2]), accel_roll, label="accelerometer roll") 
    plt.plot(np.arange(data.shape[2]), true_roll, label= "ground truth roll")
    plt.legend(loc="upper left")
    plt.subplot(2,1,2)
    plt.plot(np.arange(data.shape[2]), accel_pitch, label= "accelerometer pitch") 
    plt.plot(np.arange(data.shape[2]), true_pitch, label= "ground truth pitch")
    plt.legend(loc="upper left")
    plt.show()
    '''
    
    ###### STATE VECTOR INITIALIZE ######
    q = Quaternion()        #initial value
    q_copy = q              #declared for safety purpose
    #omega = gyro_value[:,0] #initial value
    omega = np.zeros((3,)) #debugging initial value
    omega_copy = omega      #declared for safety purpose
    #x = np.vstack(q,omega)
    #print(x) 

    n = accel_value.shape[0]+gyro_value.shape[0]
    #print(n) it is 6

    #x_hat = np.array([1,0.6,0.5,1,0.8,0.3]) #mean initialised            
    P = np.identity(n)*0.1                  #covariance matrix initialised
    Q = np.identity(n)*0.1                  #process noise
    R = np.identity(n)*5                    #measurement noise

    roll = []
    pitch = []
    yaw = []
    x_hat_list = []
    P_list = []

    ###### LOOP ######
    ###### in this outer loop, q_copy, omega_copy, x_hat and P get updated ######
    for iteration in range(T-1):          #we have lesser vicon data than imu data
        T1 = np.linalg.cholesky(n*(P+Q))
        T2 = -np.linalg.cholesky(n*(P+Q))
        
        W = np.hstack((T1, T2))
        #print(W)
        
        #print(W[:3,0])
        
        ###### GATHERING VALUES FOR PLOTTING ######
        
        #x_hat_list.append(x_hat)
        P_list.append(P)
        r,p,y = q_copy.euler_angles()
        roll.append(r)
        pitch.append(p)
        yaw.append(y)
        
        ###### CALCULATING X, Y AND Z ######
        
        X_quat = []
        X_omega = []
        
        for j in range(2*n):
            
            #calculating qw
            q_w = Quaternion()
            q_w.from_axis_angle(W[:3,j])
            #angle_w = np.linalg.norm(W[:3,j])
            #axis_w = np.divide(W[:3,j],np.linalg.norm(W[:3,j]))
            #q_w = Quaternion(np.cos(angle_w/2),np.dot(axis_w,np.sin(angle_w/2)))
            
            #calculating omega_w
            omega_w = W[3:6,j]
        
            #calculating quaternion part of sigma point
            X_quat.append(q_copy.__mul__(q_w))
            
            #calculating omega part of sigma point
            X_omega.append(omega_copy + omega_w)
        
        #X_quat is 4x12, X_omega is 3x12.  Effectively X (sigma points) dimension is 7x12 
        X_quat = np.array(X_quat)
        X_omega = np.transpose(np.array(X_omega))
        
        #print(X_quat[0])
        #print(X_quat[0].scalar())
        #print(X_omega)
        #print(X_omega[:,3])
        #print(X_omega.shape)
        
        #print(imu['ts'][0][5])
        
        #calculating delta q
        del_t = imu['ts'][0][iteration+1] - imu['ts'][0][iteration]
        del_q = Quaternion()
        del_q.from_axis_angle(omega_copy*del_t)
        
        #projecting sigma points AHEAD IN TIME 
        #Y_quat is 4x12, Y_omega is 3x12.  Effectively Y (projected sigma points) dimension is 7x12 
        
        #quaternion part gets multiplied by del_q
        Y_quat = []
        
        for t in range(2*n):
            Y_quat.append(X_quat[t].__mul__(del_q))
        
        #omega part remains same as X_omega
        Y_omega = X_omega

        #print(Y_quat[10])
        #print(Y_quat[0].scalar())
        
        #making a quaternion for only accelerometer readings
        g_vec = np.array([0,0,9.81])
        g_Q = Quaternion(0, g_vec)
        
        #quaternion part gets like q_inv.g.q type and then just the vec part is taken
        Z_accel = []
        
        for p in range(2*n):
            # no need to convert to axis angle. rather we just need the vector part of the quaternion
            Z_accel.append(((X_quat[p].inv()).__mul__(g_Q)).__mul__(X_quat[p]).vec())         
         
        Z_accel = np.transpose(np.array(Z_accel))
        #print(Z_accel.shape)
        
        #omega part remains same as X_omega
        Z_gyro = X_omega
        #print(Z_gyro.shape)
        
        Z = np.vstack((Z_accel,Z_gyro))
        #print(Z)
        #print(Z.shape)
        
        ###### COMPUTING MEAN ######
        
        #Applying gradient descent for finding mean of quaternion part (Y_quat)
        q_bar = Quaternion()
        e_vec = []
        
        while(1): 
            e_mean = np.zeros((3,)) 
            q_inv = q_bar.inv()  
            
            for k in range(2*n):
                e = Y_quat[k].__mul__(q_inv)
                e_mean += e.axis_angle()
            e_mean = e_mean/(2*n)
            #print(e_mean)
            
            if(np.linalg.norm(e_mean) <0.0001):
                for b in range(2*n):
                    e_take = Y_quat[b].__mul__(q_inv)
                    e_vec.append(e_take.axis_angle())     #converted to axis angle form       #save last e_vec for future usage
                break
            
            #print(e_vec)
            e_mean_quat = Quaternion()
            e_mean_quat.from_axis_angle(e_mean)
            q_bar = e_mean_quat.__mul__(q_bar) 
        
        #The shape of Y_quat_mean will be 4x1 as it is converted to axis angle form to become 3x1
        Y_quat_mean = q_bar.axis_angle()                
        #print(Y_quat_mean.shape)
        
        #Calculating the mean of Y_omega part using simple barycentric mean
        #The shape of Y_omega_mean will be 3x1
        Y_omega_mean = np.mean(Y_omega, axis = 1).reshape(3,)
        #print(Y_omega_mean)
        #print(Y_omega_mean.shape)
        
        #Now x_hat_bar consists of Y_quat_mean and Y_omega_mean
        #Effectively the shape of x_hat_bar (the temporary mean before x_hat is calculated) will be 6x1
        x_hat_bar = np.hstack((Y_quat_mean, Y_omega_mean))
        #print(x_hat_bar)
        #print(x_hat_bar.shape)
        
        #DETOUR : COMPUTING z_bar
        z_bar = np.mean(Z, axis = 1)
        #print(z_bar.shape)
        
        ###### COMPUTING COVARIANCE ######
        
        ###### A PRIORI STATE VECTOR COVARIANCE P_BAR ######
        
        #print(e_vec)
        r_W_dash = np.transpose(np.array(e_vec))
        omega_W_dash = []
        for d in range(2*n):
            out_of_names = Y_omega[:,d] - Y_omega_mean
            omega_W_dash.append(out_of_names)
        
        omega_W_dash = np.transpose(np.array(omega_W_dash))
        #print(omega_W_dash.shape)
        W_dash = np.vstack((r_W_dash, omega_W_dash))
        #print(W_dash)

        P_bar = np.matmul(W_dash, np.transpose(W_dash))/(2*n)      
        #print(P_bar)
        #print(P_bar.shape)     # 6x6
        
        ###### MEASUREMENT ESTIMATE COVARIANCE PZZ AND NOISY COVARIANCE PVV ######
        
        Z_minus_z_bar = []
        for rai in range(2*n):
            out_of_names_again = Z[:,rai] - z_bar
            Z_minus_z_bar.append(out_of_names_again)
        
        Z_minus_z_bar = np.transpose(np.array(Z_minus_z_bar))
        P_zz = np.matmul(Z_minus_z_bar,np.transpose(Z_minus_z_bar))/(2*n)  
        #print(P_zz)
        #print(P_zz.shape)
        
        P_vv = P_zz + R
        #print(P_vv)
        #print(P_vv.shape)
        
        ###### CROSS CORRELATION MATRIX PXZ ######
        
        P_xz = np.matmul(W_dash,np.transpose(Z_minus_z_bar))/(2*n)
        #print(P_xz)
        #print(P_xz.shape)
        
        ###### KALMAN GAIN AND UPDATE EQUATIONS ######
        
        #Kalman gain
        K = np.matmul(P_xz,np.linalg.inv(P_vv))
        #print(K)
        #print(K.shape)
        
        z = obs[:,iteration]
        innovation = z - z_bar
        #print(innovation)
        
        #Update final mean x_hat
        x_hat = x_hat_bar + np.matmul(K, innovation)
        #print(x_hat)
        
        #Updating final covariance P
        P = P_bar - np.matmul(K,np.matmul(P_vv, np.transpose(K)))
        #print(P)
        
        #Updating q_copy and omega_copy
        
        vec = x_hat[:3]
        q_end = Quaternion()
        q_end.from_axis_angle(vec)
        q_copy = q_end
        #print(q_copy)
        
        omega_copy = x_hat[3:n]
        #print(omega_copy)
        
    x_hat_list = np.array(x_hat_list)
    P_list = np.array(P_list)
    roll = np.array(roll)
    pitch = np.array(pitch)
    yaw = np.array(yaw)

    #print(roll.shape)
    
    ###### PLOTTING ALL OF THIS ######
    '''
    fig, ax = plt.subplots(3, 1, figsize=(30,6))

    ax[0].plot(roll + P_list[:,0,0], color = 'black', linestyle = 'dashed', label = 'filtered roll + std')
    ax[0].plot(roll - P_list[:,0,0], color = 'black', linestyle = 'dashed', label = 'filtered roll - std')
    ax[0].plot(roll, color = 'black', linestyle = 'solid', label = 'filtered roll')
    ax[0].plot(true_roll, color = 'pink', linestyle = 'solid', label = 'true roll')
    ax[0].set_ylabel("Roll")
    ax[0].set_title("Vicon vs Filter")
    ax[0].legend(loc = 'upper right')
    ax[0].margins(x=0)
    ax[0].grid()

    ax[1].plot(pitch + P_list[:,1,1], color = 'black', linestyle = 'dashed', label = 'filtered pitch + std')
    ax[1].plot(pitch - P_list[:,1,1], color = 'black', linestyle = 'dashed', label = 'filtered pitch - std')
    ax[1].plot(pitch, color = 'black', linestyle = 'solid', label = 'filtered pitch')
    ax[1].plot(true_pitch, color = 'orange', linestyle = 'solid', label = 'true pitch')
    ax[1].set_ylabel("Pitch")
    ax[1].legend(loc = 'upper right')
    ax[1].margins(x=0)
    ax[1].grid()

    ax[2].plot(yaw + P_list[:,2,2], color = 'black', linestyle = 'dashed', label = 'filtered yaw + std')
    ax[2].plot(yaw - P_list[:,2,2], color = 'black', linestyle = 'dashed', label = 'filtered yaw - std')
    ax[2].plot(yaw, color = 'black', linestyle = 'solid', label = 'filtered yaw')
    ax[2].plot(true_yaw, color = 'seagreen', linestyle = 'solid', label = 'true yaw')
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("Yaw")
    ax[2].legend(loc = 'upper right')
    ax[2].margins(x=0)
    ax[2].grid()

    ax[0].plot(x_hat_list[:,3] + P_list[:,3,3], color = 'black', linestyle = 'dashed', label = 'filtered gyro x + std')
    ax[0].plot(x_hat_list[:,3] - P_list[:,3,3], color = 'black', linestyle = 'dashed', label = 'filtered gyro x - std')
    ax[0].plot(x_hat_list[:,3], color = 'black', linestyle = 'solid', label = 'filtered gyro x')
    ax[0].plot(gyro_value[0,:], color = 'purple', linestyle = 'solid', label = 'true gyro x')
    ax[0].set_ylabel("Angular Velocity x")
    ax[0].set_title(" Gyro vs Filter")
    ax[0].legend(loc = 'upper right')
    ax[0].margins(x=0)
    ax[0].grid()

    ax[1].plot(x_hat_list[:,4] + P_list[:,4,4], color = 'black', linestyle = 'dashed', label = 'filtered gyro y + std')
    ax[1].plot(x_hat_list[:,4] - P_list[:,4,4], color = 'black', linestyle = 'dashed', label = 'filtered gyro y - std')
    ax[1].plot(x_hat_list[:,4], color = 'black', linestyle = 'solid', label = 'filtered gyro y')
    ax[1].plot(gyro_value[1,:], color = 'blue', linestyle = 'solid', label = 'true gyro y')
    ax[1].set_ylabel("Angular Velocity y")
    ax[1].legend(loc = 'upper right')
    ax[1].margins(x=0)
    ax[1].grid()

    ax[2].plot(x_hat_list[:,5] + P_list[:,5,5], color = 'black', linestyle = 'dashed', label = 'filtered gyro z + std')
    ax[2].plot(x_hat_list[:,5] - P_list[:,5,5], color = 'black', linestyle = 'dashed', label = 'filtered gyro z - std')
    ax[2].plot(x_hat_list[:,5], color = 'black', linestyle = 'solid', label = 'filtered gyro z')
    ax[2].plot(gyro_value[2,:], color = 'red', linestyle = 'solid', label = 'true gyro z')
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("Angular Velocity z")
    ax[2].legend(loc = 'upper right')
    ax[2].margins(x=0)
    ax[2].grid()
    '''
    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw

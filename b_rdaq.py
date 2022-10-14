
import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp



np.random.seed(10)

d=512
n=10
I=10
B = 1 #boosting parameter
print('Clients: {}, dimension: {}, Iteration: {}'.format(n,d, I))
e=0
M=[0]*4  ## 4 is enough to cover unity norms
for i in range(4):
    M[i]=min(1,np.sqrt(3/d*np.exp(e)))    
    e=np.exp(e)
M=[0.0]+M    
print('Communication per dim: B-RDAQ =',int(4*np.log2(B+1))+2)  # as per the paper; 2 counts for log(len(M))


def B_RDAQ(flat_grad, side_inf, diagonal, Bnd):
    #Random Rotation    
    flat_grad=np.multiply(diagonal, flat_grad)        
    #print(flat_grad)
    rotated_flat_grad=sp.fwht(flat_grad)
    rotated_flat_grad = np.array(rotated_flat_grad, dtype=np.float64)
    rotated_norm_grad=rotated_flat_grad/d**0.5
    #grad_norm=LA.norm(rotated_norm_grad)
    
    
    #Normalizingthegradient
    rotated_norm_grad=rotated_norm_grad/Bnd
    
    # Rotate and normalize the side-information
    side_inf=np.multiply(diagonal, side_inf)
    rotated_side_inf=sp.fwht(side_inf)
    rotated_side_inf=np.array(rotated_side_inf, dtype=np.float64)
    rotated_norm_side=rotated_side_inf/d**0.5
    #grad_norm_side = LA.norm(rotated_norm_side)
    rotated_norm_side /=Bnd 
  
	
    #Nonuniform_Level
    temp_M=np.digitize(np.absolute(rotated_norm_grad), M)
    temp_M_2=np.digitize(np.absolute(rotated_norm_side), M)
    temp_M=np.maximum(temp_M, temp_M_2)
   
    temp_2=[0.0]*d
    temp_2_s=[0.0]*d
    
    for i in range(d):
        temp_2[i]=rotated_norm_grad[i]  
        temp_2_s[i]=rotated_norm_side[i]       
   
    output=np.array([0.0]*d)
    for w in range(B):
        U_random = np.random.rand(d)
        b_temp_2=UnifQ(temp_2, U_random, temp_M)         # Use of indicators to find unbiased estimates
        b_temp_2_s=UnifQ(temp_2_s, U_random, temp_M)
    #print(temp_2, temp_2_s)
        otp=[0.0]*d
        cd=[0.0]*d
        for i in range(d):
        
            otp[i]=(b_temp_2[i]-b_temp_2_s[i])*2*M[temp_M[i]]
            cd[i] = b_temp_2[i]-b_temp_2_s[i]
            otp[i]+=rotated_norm_side[i]
        output+=np.array(otp)    
    
    output = output*Bnd/B # Averaging out the boosting effect
    
    #Inverse rotation
    output = output*(d**0.5)
    output = list(output)
    output_v=np.array(sp.ifwht(output), dtype=np.float64)
    output_v=np.multiply(output_v, diagonal)

    return  output_v
    


    
    
def UnifQ(input_v, U_r, tmp):
    U    = [2*M[tmp[i]]*(U_r[i]-0.5) for i in range(d)]    #### generates uniform random variable in [-M,M]
    #print(U)
    bits = np.array([np.where(U[i]<=input_v[i], 1.,0.) for i in range(d)])
    return bits

''' Note: In the following, we take sigma_md as our tuning parameter. Its is realted as delta_prime/8 to our paper description. '''

sigma_range=[0.000078125*(2**i) for i in range(8)]  ### Uncomment it for 6 and 8  bit comparisons
#sigma_range=[0.000078125*(2**(i-2)) for i in range(8)]  ## uncomment it only for 10 bit comparisons

MSE_B_RDAQ=[]
for sig in sigma_range:
    sigma_md= sig
    Bnd = (1+4*sig)*np.sqrt(d)
    MSE_B_RDAQ_I=[0.0]*I
    for j in range(I):
        frac=int(d**((j-1)/10.0))
        input_avg=[0.0]*d
        output_avg=[0.0]*d 
        output_avg_ratq=[0.0]*d
        output_avg_rmq=[0.0]*d
        norm_ip=0.0
        mean = np.random.rand(d)
        for client in range(n):
             diagonal=2*np.random.binomial(1, 0.5, d)-[1.0]*d    #### independent rotation  for each client
             
             t = 8*sigma_md*(np.random.rand(d)-0.5)    #### independent data for each client though closer by metric: \sigma_md
             x= mean+t
             
             y = mean+8*sigma_md*(np.random.rand(d)-0.5) 
             res = max(abs(y-x))
            
             
             Output_V=B_RDAQ(x, y, diagonal, Bnd)           
             input_avg=np.add(x, input_avg)
             
             output_avg=np.add(Output_V, output_avg)
             print('Iteration:{}, client: {}, sigma_MD: {}'.format(j, client, sigma_md))       
        MSE1=np.add(output_avg/n, -input_avg/n)
        print('Norm: TM=', LA.norm(np.array(input_avg))/n, 'B_RDAQ= ', LA.norm(np.array(output_avg))/n)
        MSE1=np.array(MSE1, dtype=np.float64)
        MSE_B_RDAQ_I[j]=LA.norm(MSE1)
        
       
   
    print('Average RMSE over iterations: RDAQ', np.mean(MSE_B_RDAQ_I))
    MSE_B_RDAQ.append(np.mean(MSE_B_RDAQ_I)) 
 
font = {'family': 'normal', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)


np.savetxt("RMSE_B_RDAQ_"+str(int(4*np.log2(B+1))+2)+"_"+str(d)+".dat", MSE_B_RDAQ, delimiter =", ", fmt ='% s')

plt.plot(sigma_range, MSE_B_RDAQ, label='B_RDAQ '+str(int(4*np.log2(B+1))+2)+'-bits')
plt.legend()
plt.xlabel('sigma_MD')
plt.ylabel('RMSE')
plt.xticks(ticks= sigma_range, labels=sigma_range)
plt.title('n='+str(n)+' d='+str(d)+' iter='+str(I))
plt.grid()
plt.show() 

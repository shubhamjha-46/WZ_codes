
import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp



np.random.seed(10)
d=1024 # dimensions
n=10 # clients
I=10 # iterations
print('Clients: {}, dimension: {}, Iteration: {}'.format(n,d, I))
e=0
M=[0]*4   # tetra-iteration levels
for i in range(4):
    M[i]=min(1,np.sqrt(3/d*np.exp(e)))    
    e=np.exp(e)

M=[0.0]+M    
K=4            #No. of bits used after finding the smallest tetra iterated interval

print('Communication per dim: RATQ=', K+2 )  # 2-bits are for log(len(M))



def RATQ(flat_grad, diagonal, U_random, Bnd):
    #Random Rotation    
    flat_grad=np.multiply(diagonal, flat_grad)        
    rotated_flat_grad=sp.fwht(flat_grad)  
    rotated_flat_grad = np.array(rotated_flat_grad, dtype=np.float64)
    rotated_norm_grad=rotated_flat_grad/d**0.5
    
    #Normalizingthegradient
    #grad_norm=LA.norm(rotated_norm_grad)
    rotated_norm_grad=rotated_norm_grad/Bnd
    #print(grad_norm)
    #Finding the smallest level in vector M
    temp_M=np.digitize( np.absolute(rotated_norm_grad), M)
    temp_2=[0.0]*d
    for i in range(d):
        temp_2[i]=rotated_norm_grad[i]
    
    #Uniform quantization in smallest level found just above
    output=UnifQ_R(temp_2, U_random, temp_M)    
    output = Bnd*output*(d**0.5)
    output = list(output)
    output_v=np.array(sp.ifwht(output), dtype=np.float64)
    output_v=np.multiply(output_v, diagonal)
    return  output_v

def UnifQ_R(input_v, U_r, tmp): #assigning to one of the 2**K levels stochastically
    a=[0.0]*d
    b=[0.0]*d
    for i in range(d):
        a[i], b[i] =M_unif(M[tmp[i]], input_v[i])
    
    q=[np.where((b[i]-a[i])*(U_r[i])< input_v[i]+a[i], b[i], a[i]) for i in range(d)]  
    q=np.array(q)   
    return q
    
def M_unif(l, inp):
	E=[-l+j*2*l/(2**K-1) for j in range(2**K)]
	f= np.digitize(inp, E)
	return E[f-1], E[f]

''' Note: In the following, we take sigma_md as our tuning parameter. Its is realted as delta_prime/8 to our paper description. '''

sigma_range=[0.000078125*(2**i) for i in range(8)]  ### Uncomment it for 6 and 8  bit comparisons
#sigma_range=[0.000078125*(2**(i-2)) for i in range(8)]  ## uncomment it only for 10 bit comparisons
MSE_RATQ=[]

for sig in sigma_range:
    sigma_md= sig
    #print('Bound', (1+4*sig)*np.sqrt(d))
    Bnd = (1+4*sig)*np.sqrt(d)
    MSE_RATQ_I=[0.0]*I
    for j in range(I):
        frac=int(d**((j-1)/10.0))
        input_avg=[0.0]*d  
        output_avg_ratq=[0.0]*d
        norm_ip=0.0
        mean = np.random.rand(d)
        for client in range(n):
			 # For independent rotation of each client's data
             diagonal=2*np.random.binomial(1, 0.5, d)-[1.0]*d    
             
             # Sampling of inputs
             t = 8*sigma_md*(np.random.rand(d)-0.5)    #### 
             x= mean+t
       
             U_random = np.random.rand(d)           
             Output_V_Ratq=RATQ(x,  diagonal, U_random, Bnd)
     
             input_avg=np.add(x, input_avg)
             output_avg_ratq=np.add(Output_V_Ratq, output_avg_ratq)
    
             print('Iteration:{}, client: {}, sigma_MD: {}'.format(j, client, sigma_md))       
       
        MSE2=np.add(output_avg_ratq/n, -input_avg/n)
        print('Norm: TM=', LA.norm(np.array(input_avg))/n, 'RATQ= ', LA.norm(np.array(output_avg_ratq))/n)
        
        MSE2=np.array(MSE2, dtype=np.float64)
        MSE_RATQ_I[j]=LA.norm(MSE2)
  
    print('Average RMSE over iterations: RATQ', np.mean(MSE_RATQ_I))
    MSE_RATQ.append(np.mean(MSE_RATQ_I)) 
   
font = {'family': 'normal', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

np.savetxt("RMSE_RATQ_"+str(K+2)+"_"+str(d)+".dat", MSE_RATQ, delimiter =", ", fmt ='% s')
plt.plot(sigma_range, MSE_RATQ, label='RATQ '+str(K+2)+'-bits')

plt.legend()
plt.xlabel('sigma_MD')
plt.ylabel('RMSE')
plt.xticks(ticks= sigma_range, labels=sigma_range)
plt.title('n='+str(n)+' d='+str(d)+' iter='+str(I))
plt.grid()
plt.show() 

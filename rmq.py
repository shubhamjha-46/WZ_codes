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
K=10            #No. of bits used for RMQ


print('Communication per dim: RMQ =', K)  
### Unif_K=[-1., 1.]


def RMQ(x,y, diagonal, U_rand, res):
	eps = 4*res/(2**(K)-2)  # as per the paper
	flat_x=np.multiply(diagonal, x)        
	
	# rotation of input
	rotated_x=sp.fwht(flat_x)
	rotated_x = np.array(rotated_x, dtype=np.float64)
	rotated_norm_x=rotated_x/d**0.5
    
    # Rotation of side-information
	flat_y=np.multiply(diagonal, y)
	rotated_y=sp.fwht(flat_y)
	rotated_y=np.array(rotated_y, dtype=np.float64)
	rotated_norm_y=rotated_y/d**0.5
   
   # Encoder uses only x, K, U_rand, eps
	z = rotated_norm_x/eps
	z=list(z)
	z_l=[int(z[i]) for i in range(d)]
	bt=[np.where(U_rand[i]<= z[i]+z_l[i], z_l[i]+1, z_l[i]) for i in range(d)]  
	bt_encoded =[int(i)%(2**(K)) for i in bt]
	
	## Decoder uses only y, K, eps, bt_encoded
	output=[0.0]*d
	W=[]
	for i in range(d):
		dist1=1
		dist2=0
		
		w=4500  ### Set 1500 for 6 bits; 4500 for 10 bits. Though, can be optimized.
		'''For smaller delta, search space needs to be large to find argmin in RMQ decoder. 
	    One way to see- smaller delta->small epsilon->more number of K-size windows. 
		Search space should be proportional to multiplicative increment in delta.'''
		while dist1>=dist2:
			w-=1
			#print(eps)
			#print((rotated_norm_y[i]-2**(K)*eps), (rotated_norm_y[i]+2**(K)*eps))
			dist1 = np.abs((bt_encoded[i]+w*2**(K))*eps-rotated_norm_y[i])
			dist2 = np.abs((bt_encoded[i]+(w-1)*2**(K))*eps-rotated_norm_y[i])
		W.append(w)
		output[i] = (bt_encoded[i]+w*2**(K))*eps
		
	# Inverse rotation
	print('W=',max(W))

	output = np.array(output)*(d**0.5)
	output = list(output)
	output_v=np.array(sp.ifwht(output), dtype=np.float64)
	output_v=np.multiply(output_v, diagonal)
	return  output_v


''' Note: In the following, we take sigma_md as our tuning parameter. Its is realted as delta_prime/8 to our paper description. '''

#sigma_range=[0.000078125*(2**i) for i in range(1)]  ### Uncomment it for 6 and 8  bit comparisons
sigma_range=[0.000078125*(2**(i-2)) for i in range(1)]  ## uncomment it only for 10 bit comparisons

MSE_RMQ=[]
for sig in sigma_range:
    sigma_md= sig
    res = 8*sig
    MSE_RMQ_I=[0.0]*I
    for j in range(I):
        frac=int(d**((j-1)/10.0))
        input_avg=[0.0]*d
        output_avg_rmq=[0.0]*d
        norm_ip=0.0
        mean = np.random.rand(d)
        for client in range(n):
             diagonal=2*np.random.binomial(1, 0.5, d)-[1.0]*d    #### independent rotation  for each client
             
             t = 8*sigma_md*(np.random.rand(d)-0.5)   
             x= mean+t  # sampling the input
             y = mean+8*sigma_md*(np.random.rand(d)-0.5) # side-information
             
             #res = max(abs(y-x))   # required for choosing parameter epsilon in RMQ
             
             U_random = np.random.rand(d)
            
             Output_V_Rmq=RMQ(x,y, diagonal, U_random, res)
            
             input_avg=np.add(x, input_avg)
            
             output_avg_rmq=np.add(Output_V_Rmq, output_avg_rmq)
             print('Iteration:{}, client: {}, sigma_MD: {}'.format(j, client, sigma_md))       
        
        MSE3=np.add(output_avg_rmq/n, -input_avg/n)
        print('Norm: TM=', LA.norm(np.array(input_avg))/n, 'RMQ= ', LA.norm(np.array(output_avg_rmq))/n)
        MSE3=np.array(MSE3, dtype=np.float64)
        MSE_RMQ_I[j]=LA.norm(MSE3)

    print('Average RMSE over iterations: RMQ', np.mean(MSE_RMQ_I))
    MSE_RMQ.append(np.mean(MSE_RMQ_I)) 

font = {'family': 'normal', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

#np.savetxt("RMSE_RMQ_"+str(K)+"_"+str(d)+".dat", MSE_RMQ, delimiter =", ", fmt ='% s')
plt.plot(sigma_range, MSE_RMQ, label='RMQ '+str(K)+'-bits')

plt.legend()
plt.xlabel('sigma_MD')
plt.ylabel('RMSE')
plt.xticks(ticks= sigma_range, labels=sigma_range)
plt.title('n='+str(n)+' d='+str(d)+' iter='+str(I))
plt.grid()
plt.show() 

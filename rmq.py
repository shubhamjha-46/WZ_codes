import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
import math


np.random.seed(10) 

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
	bt=[np.where(U_rand[i]<= z[i]-z_l[i], z_l[i]+1, z_l[i]) for i in range(d)]  
	bt_encoded =[int(i)%(2**(K)) for i in bt]
	
	## Decoder uses only y, K, eps, bt_encoded
	output=[0.0]*d
	W=[]
	for i in range(d):
		m1 = math.floor((rotated_norm_y[i]/eps+2**K-bt_encoded[i])/2**K)
		m2 = math.floor((rotated_norm_y[i]/eps-2**K-bt_encoded[i])/2**K)
		#print(m1,m2)
		t_var = m2
		chc = []
		while t_var<=m1:
			chc.append(np.abs((t_var*(2**K)+bt_encoded[i])*eps-rotated_norm_y[i]))
			t_var+=1
		w	=	np.argmin(chc)+m2
		output[i] = (bt_encoded[i]+w*2**(K))*eps
		#### Brute-force method #####
		'''For smaller delta, search space needs to be large to find argmin in RMQ decoder. 
	    One way to see- smaller delta->small epsilon->more number of K-size windows. 
		Search space should be proportional to multiplicative increment in delta.'''
		
		'''dist1=1
		dist2=0
		w=4500  ### Set 1500 for 6 bits; 4500 for 10 bits. Though, can be optimized.
		while dist1>=dist2:
			w-=1
			dist1 = np.abs((bt_encoded[i]+w*2**(K))*eps-rotated_norm_y[i])
			dist2 = np.abs((bt_encoded[i]+(w-1)*2**(K))*eps-rotated_norm_y[i])
		W.append(w)
		#if (np.argmin(chc)+m2==w):
		#	print('true')'''

	####Inverse rotation######
	output = np.array(output)*(d**0.5)
	output = list(output)
	output_v=np.array(sp.ifwht(output), dtype=np.float64)
	output_v=np.multiply(output_v, diagonal)
	return  output_v


''' 	1. Note: In the following, we take sigma_md as our tuning parameter. Its is realted as delta_prime/8 to our paper description. 
	2. We are using different values of x axis for 6 and 10 bits.'''
	
d = 512 # dimensions
I = 20 # iterations
e = 0
M = [0]*4   # tetra-iteration levels
for i in range(4):
    M[i] = min(1,np.sqrt(3/d*np.exp(e)))    
    e = np.exp(e)

M = [0.0]+M    
K = 6          #No. of bits used for RMQ
print('Communication per dim: RMQ =', K)  


for n in [5,10,20,30,40,50]:
	print('Clients: {}, dimension: {}, Iteration: {}'.format(n,d, I))
	sigma_range = [0.0125/16]  ### Only for comparison varying clients
	#sigma_range = [0.000078125*(2**(i-2)) for i in range(8)]  ### Uncomment it for 6 and 8  bit comparisons
	#sigma_range=[0.000078125*(2**(i-4)) for i in range(8)]  ## uncomment it only for 10 bit comparisons

	MSE_RMQ = []
	for sig in sigma_range:
		sigma_md = sig
		res = 8*sig
		MSE_RMQ_I = [0.0]*I
		for j in range(I):
			frac = int(d**((j-1)/10.0))
			input_avg = [0.0]*d
			output_avg_rmq = [0.0]*d
			norm_ip = 0.0
			mean = np.random.rand(d)
			for client in range(n):
				 diagonal = 2*np.random.binomial(1, 0.5, d)-[1.0]*d    #### independent rotation  for each client
				 
				 t = 8*sigma_md*(np.random.rand(d)-0.5)   
				 x= mean+t  # sampling the input
				 y = mean+8*sigma_md*(np.random.rand(d)-0.5) # side-information                          
				 U_random = np.random.rand(d)
				
				 Output_V_Rmq = RMQ(x,y, diagonal, U_random, res)
				
				 input_avg = np.add(x, input_avg)
				
				 output_avg_rmq = np.add(Output_V_Rmq, output_avg_rmq)
				 print('Iteration:{}, client: {}, sigma_MD: {}'.format(j, client, sigma_md))       
			
			MSE3 = np.add(output_avg_rmq/n, -input_avg/n)
			print('Norm: TM=', LA.norm(np.array(input_avg))/n, 'RMQ= ', LA.norm(np.array(output_avg_rmq))/n)
			MSE3 = np.array(MSE3, dtype=np.float64)
			MSE_RMQ_I[j] = LA.norm(MSE3)

		print('Average RMSE over iterations: RMQ', np.mean(MSE_RMQ_I))
		MSE_RMQ.append(np.mean(MSE_RMQ_I)) 

	font = {'weight': 'normal', 'size': 14}
	matplotlib.rc('font', **font)

	np.savetxt('n='+str(n)+"_RMSE_RMQ_"+str(K)+"_"+str(d)+".dat", MSE_RMQ, delimiter =", ", fmt ='% s')
	plt.plot(sigma_range, MSE_RMQ, label='RMQ '+str(K)+'-bits')

plt.legend()
plt.xlabel('sigma_MD')
plt.ylabel('RMSE')
plt.yscale('log')
plt.xscale('log')
plt.yticks(fontsize=7)
plt.xticks(ticks= sigma_range, labels=sigma_range, fontsize=7)
plt.title('n='+str(n)+' d='+str(d)+' iter='+str(I))
plt.grid()
plt.show()

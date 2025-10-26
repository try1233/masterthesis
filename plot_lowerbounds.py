from src.utils import *
import matplotlib.pyplot as plt

def ploting_lines(image, sticker_size,sticker_norm,hparams):

    delta = torch.zeros(image.shape)
    delta[:,0:sticker_size,0:sticker_size] = 1
   
    norm_norm = torch.linalg.norm(delta)
    delta = sticker_norm/norm_norm * delta
    delta_value = delta[:,0,0].sum().item()
    window_size = hparams['smoothing_config']['window_size']
    sigma = hparams['smoothing_config']['std']

    delta_block = min(1,(window_size + sticker_size -1) **2 / (delta.shape[1]*delta.shape[2]))
    p_x_r1 = prob_func(1,delta, window_size, sigma, sign=1)
    p_x_r2 = 1 - delta_block
    p_x_r3 = prob_func(1,delta, window_size, sigma, sign=-1)


    x_values =[i/10000 for i in range(10001)]
    worst_case,old_worst_case = generate_line_worstcases(delta, delta_block ,window_size, sigma)
    if hparams['smoothing_config']['noise_type']=="gaussian":
        values_cohen = generate_line_cohen(delta, sigma)
       
        plt.plot(x_values, worst_case, label=r'$p_{\tilde{X},y}$ ours, worst case')
        plt.plot(x_values, values_cohen, label=r'$p_{\tilde{X},y}$ Randomized Smoothing')
    elif  hparams['smoothing_config']['noise_type']=="uniform":
        values_gamma = generate_line_gamma(delta,delta_value, window_size,sticker_size=sticker_size, gamma = sigma)
        plt.plot(x_values, old_worst_case, label='$p_{x,y} - \Delta$') 
        plt.plot(x_values, values_gamma, label='ours, uniform noise', color = 'red')
    #values = generate_line(p_x_r1,p_x_r2,p_x_r3, delta, block_size, sigma)

    
   
    
   
    

    plt.plot(x_values, x_values, label='$p_{x,y}$')
    #plt.plot(x_values, values, label='$p_{\tilde{X},y}$ uniform delta distr.')
    
   
   
    
    plt.axhline(y= 0.5, color='r', linestyle='--', label='0.5')
    plt.xlim(0,1.1)
    plt.legend(title = f'$\sigma={sigma},$ $\|\delta\|_2={sticker_norm}$')
    plt.savefig('lower_bounds_patch.png') 
    plt.show()
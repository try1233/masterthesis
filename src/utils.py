import numpy as np
import math
import scipy.stats as stats
from scipy.stats import norm
import torch

def prob_func(k,delta,block_size, sigma,sign = 1):
    sum = 0
    delta_index = 0
    M = delta.shape[1] * delta.shape[2] #M is number of possible windows we take a look at
    for xcorner in range(delta.shape[1]):
        for ycorner in range(delta.shape[2]):
            out_c1 = torch.zeros(delta.shape)#.to(device)
            if (xcorner + block_size > delta.shape[1]):
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = delta[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:, :ycorner + block_size - delta.shape[2]]

                    out_c1[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]] = \
                        delta[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = delta[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = delta[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        delta[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
            out_c1 = out_c1.permute(1,2,0)
             
            delta_m = np.linalg.norm(out_c1)
            delta_index += delta_m
            if delta_m >0:
                sum += 1/M * stats.norm.cdf(sigma * math.log(k)/delta_m + sign * delta_m/(2 * sigma))
            
    return sum#s‚, delta_index


def distr(delta,block_size,sign = 1):
    sum = 0
    delta_index = 0
    distr = {}
   
    for xcorner in range(delta.shape[1]):
        for ycorner in range(delta.shape[2]):
            out_c1 = torch.zeros(delta.shape)#.to(device)
            if (xcorner + block_size > delta.shape[1]):
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = delta[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:, :ycorner + block_size - delta.shape[2]]

                    out_c1[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]] = \
                        delta[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = delta[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = delta[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        delta[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
            out_c1 = out_c1.permute(1,2,0)
             
            summe = torch.sum(out_c1).item()
            if summe in distr:
                distr[summe] +=1
            else: distr[summe] = 1
            
    return distr#s‚, delta_index

def prob_func_unif(input,delta_value,block_size, sticker_size, gamma,sign = 1):
    sum = 0
    s_2 = block_size ** 2
    delta = torch.zeros(input.shape)
    delta[:,0:sticker_size,0:sticker_size] = 1
    M = delta.shape[1] * delta.shape[2] #M is number of possible windows we take a look at
    for xcorner in range(delta.shape[1]):
        for ycorner in range(delta.shape[2]):
            out_c1 = torch.zeros(delta.shape)#.to(device)
            if (xcorner + block_size > delta.shape[1]):
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = delta[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:, :ycorner + block_size - delta.shape[2]]

                    out_c1[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]] = \
                        delta[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = delta[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = delta[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        delta[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
            out_c1 = out_c1.permute(1,2,0)
            mask = out_c1 != 0
            mask = mask.float() 
            delta_m = torch.sum(mask).item()
            vol_l_2 =  max(0,(2*gamma - delta_value)) ** delta_m 
            vol_b_gamma = (1/(2*gamma)) ** delta_m
            sum += 1/M *  (1 - vol_l_2 * vol_b_gamma)
                
    return sum#s‚, delta_index




def generate_line(a, b,c, delta, block_size, sigma):
    """Generate certificate values for ablation smoothing"""
    def f(k):
        return prob_func(k,delta, block_size, sigma, sign=1) 
    def g(k):
        return prob_func(k,delta, block_size, sigma, sign=1) -  prob_func(1,delta, block_size, sigma, sign=1)

    values = []
    k_vals = []
    for x in np.arange(0,1.001,0.001):
        if x <a:
            p_val = x
        
            try:
                k_val = find_root(f, 1e-10, 1, p_val, 1e-4)
                k_vals.append(k_val)
                values.append(prob_func(k_val,delta, block_size, sigma, sign = -1))
            except:
                values.append(0)
        elif x< a + b:
            k_vals.append(1)
            values.append(x + c - a)
        else:
            p_val = (x - a - b)
            k_val = find_root(g, 1, 2**100, p_val, 1e-4)
            k_vals.append(k_val)
            value = prob_func(k_val, delta, block_size, sigma, sign = -1) - prob_func(1, delta, block_size, sigma,sign=-1 )
            values.append(value + b + c)
    return values

def generate_line_gamma(delta,delta_value, block_size,sticker_size, gamma):
    values = []
    c =  prob_func_unif(delta,delta_value, block_size,sticker_size = sticker_size,gamma = gamma)
    for x in np.arange(0,1.0001,0.0001):
        values.append(max(0,x - c))
    return values

def generate_line_cohen(delta,sigma):
    values = []
    d_norm= np.linalg.norm(delta)
    for x in np.arange(0,1.0001,0.0001):
        values.append(stats.norm.cdf(stats.norm.ppf(x) - d_norm/sigma))
    return values

def generate_line_worstcases(delta, Delta_block,block_size, sigma):
    values1 = []
    old = []
    delta_value = np.linalg.norm(delta)
    a = prob_func_wc(delta,block_size, sigma,sign = 1) 
    c = prob_func_wc(delta,block_size, sigma,sign = -1)
    for x in np.arange(0,1.0001,0.0001):
    
        old.append(max(0,x-Delta_block))#bisheriger Worst Case p_x,y - Dreieck
     
        if x <= a:
            values1.append(Delta_block * norm.cdf(norm.ppf(x/Delta_block)- delta_value/(sigma))) #überall gleiches budget
            
        elif x <= a + 1 - Delta_block:
            values1.append(x + Delta_block*(1-2*norm.cdf(delta_value/(2*sigma))))
        else:
            values1.append(1-Delta_block + Delta_block * norm.cdf(norm.ppf((x - (1-Delta_block))/Delta_block) - delta_value/(sigma)))
    return values1, old


def prob_func_wc(delta, block_size, sigma,sign = 1):
    """worst case for hierarchical randomized smoothing with patch attack"""
    sum = 0
    delta_value = np.linalg.norm(delta)
    M = delta.shape[1] * delta.shape[2] #M is number of possible windows we take a look at
    for xcorner in range(delta.shape[1]):
        for ycorner in range(delta.shape[2]):
            out_c1 = torch.zeros(delta.shape)#.to(device)
            if (xcorner + block_size > delta.shape[1]):
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = delta[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:, :ycorner + block_size - delta.shape[2]]

                    out_c1[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]] = \
                        delta[:, :xcorner + block_size - delta.shape[1], :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = delta[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size] = \
                        delta[:, :xcorner + block_size - delta.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > delta.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = delta[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]] = \
                        delta[:, xcorner:xcorner + block_size, :ycorner + block_size - delta.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        delta[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
            out_c1 = out_c1.permute(1,2,0)
             
            delta_m = np.linalg.norm(out_c1)
           
            if delta_m >0:
                sum += 1/M * stats.norm.cdf( sign * delta_value/(2 * sigma))
            
    return sum#s‚, delta_index



def find_root(f, a, b, p, tol=1e-6, max_iter=1000):
    def g(x):
        return f(x) - p
    
    if g(a) * g(b) >= 0:
        print("Error")
        return None
    
    iter_count = 0
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if g(midpoint) == 0:
            return midpoint
        elif g(a) * g(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iter_count += 1
        if iter_count >= max_iter:
            print("Max iterations")
            return None

    return (a + b) / 2

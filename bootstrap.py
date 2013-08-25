#!/usr/bin/env python
#coding:utf-8
"""
"""
import numpy as np
import numpy.random as npr
import pylab
import matplotlib.pyplot as plt

class Bootstrap:
    def __init__(self,data,num_samples,times):
        self.data=data
        self.num_samples=num_samples
        self.times=times
        self.n=data.shape[1]

    def bootstrap(self, statistic, alpha):
        idx=npr.randint(0,self.n,(self.num_samples,self.n))
        samples=self.data[idx]
        stat=np.sort(statistic(samples,1))
        return (stat[int((alpha/2.0)*self.num_samples)],
                stat[int((1-alpha)/2.0)*self.num_samples])

    def bayesian_bootstrap(self,par=1):
        weight=npr.dirichlet([par]*self.n,(self.times,self.num_samples))
        samples=[npr.choice(self.data[j],size=self.n,p=weight[j][i]) 
                for i in xrange(self.num_samples) for j in xrange(self.times)]
                #for i in xrange(self.times*self.num_samples)]
        samples=np.array(samples).reshape(self.times,self.num_samples,self.n)
        return samples

    def make_weight(self,n):
        q=np.zeros([self.times,self.num_samples,n])
        l1,l2=npr.randint(0,n,(2,self.times,self.num_samples,n))
        mu=npr.random([self.times,self.num_samples,n])
        for i in xrange(self.times):
            for j in xrange(self.num_samples):
                for l1_n,l2_n,mu_n in zip(l1[i][j],l2[i][j],mu[i][j]):
                    q[i][j][l1_n]+=mu_n/n
                    q[i][j][l2_n]+=(1-mu_n)/n
        return q


    def TOON_bootstrap(self, alpha=0.05):
        weight=self.make_weight(self.n)
        samples=[npr.choice(self.data[j],size=self.n,p=weight[j][i])
                for i in xrange(self.num_samples) for j in xrange(self.times)]
        #stat=statistic(samples,1)
        #return stat
        return samples

    def continuous_bootstrap(self):
        rand_num=npr.random([self.times,self.num_samples,self.n])
        alpha=1.4715
        pade=rand_num/(alpha-rand_num)
        pade_sum=np.repeat(pade.sum(2),self.n).reshape([self.times,self.num_samples,self.n])
        weight=pade/pade_sum
        samples=[npr.choice(self.data[j],size=self.n,p=weight[j][i])
                for i in xrange(self.num_samples) for j in xrange(self.times)]
        samples=np.array(samples).reshape(self.times,self.num_samples,self.n)
        return samples

    def broken_line(self):
        data=np.sort(self.data)
        head=data[...,0]*1.5-data[...,1]*0.5
        tail=data[...,self.n-1]*1.5-data[...,self.n-2]*0.5
        data=np.c_[(head,data,tail)]
        l=npr.randint(1,self.n+1,size=(self.times,self.num_samples,self.n))
        u=npr.random([self.times,self.num_samples,self.n])
        y=np.empty([self.times,self.num_samples,self.n])
        for i in xrange(self.times):
            for j in xrange(self.num_samples):
                for k in xrange(self.n):
                    if (l[i][j][k]<self.n):
                        y[i][j][k]=data[i][l[i][j][k]]*u[i][j][k]+data[i][l[i][j][k]+1]*(1-u[i][j][k])
                    elif(npr.random()<=0.5):
                        y[i][j][k]=data[i][0]*u[i][j][k]+data[i][1]*(1-u[i][j][k])
                    else:
                        y[i][j][k]=data[i][self.n]*u[i][j][k]+data[i][self.n+1]*(1-u[i][j][k])
        y=np.array(y).reshape(self.times,self.num_samples,self.n)
        return y





def main():
    times=10
    data=npr.normal(0,1,(times,100))
    #\\print data
    num_samples=20
    test = Bootstrap(data,num_samples,times)
    #samples = test.TOON_bootstrap()
    #samples = test.continuous_bootstrap()
    #samples = np.array(samples).reshape(times,num_samples,data.shape[1])
    #samples = test.bayesian_bootstrap()
    samples = test.broken_line()
    result=np.apply_over_axes(np.mean,samples,[1,2])
    np.save("broken_line",samples)
    #np.save("continuous_bootstrap",samples)
    #np.save("toon_bootstrap",samples)
    #print samples
    print result



if __name__ == '__main__':
    main()


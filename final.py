import numpy as np  
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
TR = 0.4
TS = int(100-(TR*100))
def nonlin(x, deriv=False):  
    if(deriv==True):
        
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  
 


linked_full=pd.read_csv('data.csv')
col_names = linked_full.columns.values
linked_full.head()
train,test =train_test_split(linked_full,test_size=TR)
X=train.ix[:,(1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20)].values
Y=train.ix[:,(21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37)].values
Y=Y.reshape(TS,17)


X = np.array(X)

Y = np.array(Y)


np.random.seed(1)




#synapses
syn0 = 2*np.random.random((19,TS)) - 1  
syn1 = 2*np.random.random((TS,17)) - 1  



for j in range(60000):  
    
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    l2_error = Y - l2
    if(j % 10000) == 0:   
        print("Error: " + str(np.mean(np.abs(l2_error))))
    
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)

    
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
#print(syn1.shape)
#print(syn0.shape)
print("Output after training")
#print(l2)
sugg=[]
for i in l2:
	a=[]
	for j in i:
		j=int(str(j).split(".")[0])
		a.append(j)
	sugg.append(a)
#print(sugg)  
l1=list(map(int,input("enter values").split(" ")))

l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))
print("Output ")
#print(l2)
sugg=[]
for i in l2:
	a=[]
	for j in i:
		j=int(str(j).split(".")[0])
		a.append(j)
	sugg.append(a)
ans = sugg[-1]
#print(ans) 
#print(col_names)
d={}
print(ans)
for i in range(0,17):
	d[ans[i]]=[]
#print(d)
try:
	for i in range(19,len(col_names)+19):
		#print(col_names[i],"should be given priority",ans[i-19])
		if(i!= 19 and i!=20):
			d[ans[i-19]].append(col_names[i])
except:
	print("")
for k in sorted(d.keys()):
	print (', '.join(d[k]),end=" ")
	print("should be given priority",k)




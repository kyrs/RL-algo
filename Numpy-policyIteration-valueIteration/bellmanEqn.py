"""
__author__  : Kumar Shubham 
__date__    : 15-12-2018
__desc__    : bellman equation implementation
"""
import numpy as np 





def bellmanEqn(initialValue,utility,TranProb,reward, gamma):
    ## applying bellman eqn:
    arrList = []
    for i in range(4): # no of possible action
        out = np.dot(utility,TranProb[:,:,i].T)

        arrList.append(out)

    maxValue = (np.max(np.vstack(arrList),axis=0))
    print("maxValue",maxValue)
    output = reward+ gamma*maxValue
    print(output)

    ## current position of the element 
    print (np.sum(np.multiply(initialValue,output)))
    return 
    
def main():
    TranProb = np.load("T.npy") ## array of shape 12X12X4. with transation probability 
    #Starting state vector
    #The agent starts from (1, 1)
    v = np.array([[0.0, 0.0, 0.0, 0.0, 
                   0.0, 0.0, 0.0, 0.0, 
                   1.0, 0.0, 0.0, 0.0]]) ## shape 1X12

    ## initial value function
    u = np.array([[0.812, 0.868, 0.918,   1.0,
               0.762,   0.0, 0.660,  -1.0,
               0.705, 0.655, 0.611, 0.388]])## utility of current step 

    ## reward of moving to any adjacent box is same 
    reward = -0.04
    gamma = 1.0
    ## bellman eqn based function 
    bellmanEqn(initialValue=v,utility=u,TranProb=TranProb,reward=reward,gamma=gamma)
    

if __name__ =="__main__":
    main()
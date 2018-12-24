"""
__author__  : Kumar Shubham 
__date__    : 15-12-2018
__desc__    : value iteration implementation
"""
import numpy as np 
import copy

def valueIterationAlgo(valueFn,reward,tranProb,noAction,gamma):
    ### following function will do the value iteration based on bellman algorithm 

    # uValue    : previous value function
    # reward    : reward on reaching a given state
    # transPorb : transition probability of matrix
    # noAction  : no of defined action
    # gamma     : discount factor

    listActValFn = []
    for action in range(noAction):
        tranProbPerAct = tranProb[:,:,action]
        out = np.dot(valueFn,tranProbPerAct.T)
        listActValFn.append(np.array(out))
    maxValuePerAct=(np.max(np.vstack(listActValFn),axis=0))
    finalOut = reward+gamma*maxValuePerAct
    return finalOut
def main():
    transProb = np.load("T.npy") ## array of shape 12X12X4. with transation probability 
    #Starting state vector
    


    ## initial value function
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])## utility of current step 

    ## reward of moving to any adjacent box is same 
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])    


    gamma = 0.5
    ## bellman eqn based function 
    epsilon = 0.001
    count = 0
    oldValueFn = u.copy()

    while (True):
        ## do iteration till it converge
        count+=1
        # print(oldValueFn)
        newValueFn = valueIterationAlgo(oldValueFn,r,transProb,4,gamma)
        convergScore = np.max(np.abs(newValueFn-oldValueFn))
        # print("conv score : %f"%(convergScore))
        if (convergScore < (epsilon*(1-gamma)/gamma)):
            break
        else :
            oldValueFn = newValueFn.copy()      


    print ("==========output==========")
    print ("iteration : %d"%(count))
    print ("convergence score %f"%(convergScore))
    print ("final iteration count : %d"%(count))
    print ("=========value function====")
    print(newValueFn[0:4])
    print(newValueFn[4:8])
    print(newValueFn[8:12])

if __name__ =="__main__":
    main()
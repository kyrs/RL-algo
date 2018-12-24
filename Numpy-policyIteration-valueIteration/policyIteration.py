"""
__author__  : Kumar Shubham 
__date__    : 15-12-2018
__desc__    : value iteration implementation
"""
import numpy as np 
import copy

def greedyPolicyCalculation(valueFn,tranProb,noAction):
    ### following function will calculate the optimal policy for evaluation 

    # uValue    : previous value function
    # transPorb : transition probability of matrix
    # noAction  : no of defined action


    listActValFn = []
    for action in range(noAction):
        tranProbPerAct = tranProb[:,:,action]
        out = np.dot(valueFn,tranProbPerAct.T)
        listActValFn.append(np.array(out))
    vertStackValue = np.vstack(listActValFn)
    maxValuePerAct=np.argmax(vertStackValue,axis=0)
    return maxValuePerAct


def valueFuncCalc(valueFn,reward,tranProb,noAction,gamma,policy,stateNo):
    ### following function will calculate the value function
    ### NOTE: it is not bellman optimal eqn implementation  

    # uValue    : previous value function
    # reward    : reward on reaching a given state
    # transPorb : transition probability of matrix
    # noAction  : no of defined action
    # gamma     : discount factor
    # policy    : policy defiend in optimal policy calculation
    # stateNo   : no of state in the system

    out = np.zeros((stateNo,))
    for state in range(stateNo): 
        action = policy[state]  
        tranProbPerAct = tranProb[state,:,action]
        out[state] = np.dot(valueFn,tranProbPerAct.T)

    finalOut = reward+gamma*out
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


    gamma = 0.999
    ## bellman eqn based function 
    epsilon = 0.001
    count = 0
    oldValueFn = u.copy()

    while (count<22):
        ## do iteration till it converge
        count+=1
        # print(oldValueFn)
        PolicyCalc = greedyPolicyCalculation(valueFn=oldValueFn,tranProb=transProb,noAction=4)
        newValueFn = valueFuncCalc(valueFn=oldValueFn,reward=r,tranProb=transProb,noAction=4,gamma=gamma,policy=PolicyCalc,stateNo=12)

        convergScore = np.max(np.abs(newValueFn-oldValueFn))
        # print("conv score : %f"%(convergScore))
        if (convergScore < (epsilon*(1-gamma)/gamma)):
            #break
            pass
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


    charPolicyMap = {0:" ^ ",1:" < ", 2:" v ", 3:" > ",-1 :" * ", -2:" # "}
    print ("=========Policy===========")
    countPolicy = 0
    
    ### defining policy for goal as -1 Just for print
    PolicyCalc[3] = -1
    PolicyCalc[7] = -1
    PolicyCalc[6] = -2

    print( "".join(charPolicyMap[i] for i in PolicyCalc[0:4] ) )
    print( "".join(charPolicyMap[i] for i in PolicyCalc[4:8] ) )
    print( "".join(charPolicyMap[i] for i in PolicyCalc[8:12] ) )


if __name__ =="__main__":
    main()
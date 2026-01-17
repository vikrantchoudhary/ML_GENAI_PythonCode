# softmax for muticlass (mutually exclusive classes) classification & that can be used for self attention
import numpy as np
def softmax(vector_z):
    #subtracting np.max(z) for numerical stability
    e_z = np.exp(vector_z - np.max(vector_z,axis=-1,keepdims=True))
    p = e_z / e_z.sum(axis=-1, keepdims=True)
    return p

def softmax2(vector_z):
    p = np.exp(vector_z) / np.sum(np.exp(vector_z))
    return p

def self_attention(input_matrix,W_q,W_k,W_v): #query,key & value for self attention
    """
    Docstring for self_attention
    
    :param input_matrix: (seq_len,embedding_dim)
    :param W_q: Weight matrices for Query
    :param W_k: Key
    :param W_v: value
    """
    # get the inputs into Q,K,V spaces
    Q = np.dot(input_matrix,W_q)
    K = np.dot(input_matrix,W_k)
    V = np.dot(input_matrix,W_v)

    # calculate attention score (dot product of Q & K)
    d_k = Q.shape[-1]
    scores = np.dot(Q,K.T)/np.sqrt(d_k)

    #apply softmax to get weights attention (probablities)
    weights = softmax(scores)

    # multiply weight by value to get final output
    output = np.dot(weights,V)

    return output,weights

# Example 
seq_len,embed_dim = 3,4  # 3 words , eachi with 4-dim embedding
d_k = 2 # internal dimension of Q,K,C

#mock _data 
x = np.random.rand(seq_len,embed_dim)
print(f"Input dat {x}")
W_q = np.random.rand(embed_dim,d_k)
W_k = np.random.rand(embed_dim,d_k)
W_v = np.random.rand(embed_dim,d_k)

output,weights = self_attention(x,W_q,W_k,W_v)
print("Attention weights (propbablities) : \n ", weights)
print("\noutput shape: ", output.shape) 
print("\n output:  " , output)


'''
z = np.array([-3.0,2.0,1.0,1.6,0.2, 2.1])
print("\n",softmax2(z))
print("\n")
print(softmax(z))
'''



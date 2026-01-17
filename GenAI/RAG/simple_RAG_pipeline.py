#Given a query embedding and a database dictionary of documents embedding returns 
# most relevant document ID
#without using any predefined lib
import numpy as np

def simple_retrieval(query_vec,vector_db):
    """
    Docstring for simple_retrieval
    
    param query_vec: np.array (1D)
    param vector_db: dict {doc_id : np.array}
    """
    best_doc = None
    highest_sim = -1.0  #similarity

    for doc_id,doc_vec in vector_db.items():
        sim = np.dot(query_vec, doc_vec)
        if sim > highest_sim:
            highest_sim = sim
            best_doc = doc_id

    return best_doc,highest_sim

#Mock data
db = {
    "doc_1" : np.array([0.1,0.9,0.0]),  # TOPIC : AI
    "doc_2" : np.array([0.8,0.1,0.1])    #TOPIC : cooking
}
query = np.array([0.05,0.95,0.01]) # query about AI

doc, score = simple_retrieval(query_vec=query,vector_db=db)

print(f"Top document : {doc} ,with score {score : .4f} ")
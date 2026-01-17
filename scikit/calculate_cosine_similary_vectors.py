#using context embedding calculate cosine similarity between vectors without/with sckit-learn

import math

def cosine_similarity(v1,v2):
    dot_product = sum(a*b for a,b in zip(v1,v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(b**2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


vec1 = [1,2,3]
vec2 = [1,2,3.1]

print(f"similary : {cosine_similarity(vec1, vec2):.4f}")
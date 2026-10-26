"""Small interpretable triangular-membership fuzzy priority controller."""
import numpy as np

def tri(x,a,b,c):
    if x<=a or x>=c: return 0.
    if x==b: return 1.
    return (x-a)/(b-a) if x<b else (c-x)/(c-b)

def fuzzy_cooling_priority(temp_c, occupancy, price):
    hot=tri(temp_c,22,30,38); busy=tri(occupancy,0.15,.8,1.0); cheap=1-tri(price,5,10,15)
    # Mamdani-style rule firing strengths: hot AND busy AND cheap => strong cooling.
    strong=min(hot,busy,cheap); moderate=min(hot,busy); low=max(0,1-max(strong,moderate*.65))
    return float(np.clip((strong*1.0+moderate*.65+low*.2)/(strong+moderate+low+1e-8),0,1))

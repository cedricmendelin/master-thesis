import numpy as np 

def boolean_string(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def normalize(im):
    return (im-im.min())/(im.max()-im.min())

def clip_to_uint8(x):
    return np.clip(x*255,0,255).astype(np.uint8)
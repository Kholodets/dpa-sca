import numpy as np
ciphers = np.load("aging_trace_data/cipher_texts.npy")
for a in ciphers:
    s = "".join('{:02x}'.format(n) for n in a) 
    print("0x",s, sep = '')

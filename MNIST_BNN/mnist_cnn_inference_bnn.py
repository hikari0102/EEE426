import numpy as np
from scipy.signal import convolve2d
from tqdm import trange

mnist = np.load("mnist-original.npy", allow_pickle=True).item()
X = mnist["data"].T / 255.0        
y = mnist["label"][0]         

w = np.load("weights.npy", allow_pickle=True).item()
conv1w = np.sign(w["conv1.weight"])
conv2w = np.sign(w["conv2.weight"])
conv3w = np.sign(w["conv3.weight"])
fc_w  = np.sign(w["fc1.weight"])  # shape (10, 32*11*11)
fc_b  =      w["fc1.bias"]        # shape (10,), but we will not use bias in this inference as not using bias shown better accuracy

def binarize(x):
    return np.where(x > 0, 1.0, -1.0)

def avg_pool2d(x, kernel=2, stride=2):
    N, C, H, W = x.shape
    out_H = H // kernel
    out_W = W // kernel
    # sum pooling
    y = np.add.reduceat(
        np.add.reduceat(x, np.arange(0, H, stride), axis=2),
        np.arange(0, W, stride), axis=3
    )
    return y / (kernel*kernel)  # shape (N, C, out_H, out_W)

def feed_forward(X0):
    """
    X0: (batch_size, 784) float in [0,1]
    returns logits: (batch_size, 10)
    """
    N = X0.shape[0]
    X0 = X0.reshape(N, 1, 28, 28)
    X0 = (X0 > 0.5).astype(np.float32) 

    X1 = np.zeros((N, 16, 26, 26), dtype=np.float32)
    for b in range(N):
        for co in range(16):
            # as we using scipy's MNIST and scipy's convolve2d, we have to flip the kernel
            # to match the original PyTorch implementation
            kernel = conv1w[co, 0][::-1, ::-1]
            X1[b, co] = convolve2d(X0[b, 0], kernel, mode="valid")
    X1 = binarize(X1)

    X2 = np.zeros((N, 32, 24, 24), dtype=np.float32)
    for b in range(N):
        for co in range(32):
            acc = np.zeros((24, 24), dtype=np.float32)
            for ci in range(16):
                kernel = conv2w[co, ci][::-1, ::-1]
                acc += convolve2d(X1[b, ci], kernel, mode="valid")
            X2[b, co] = acc
    X2 = binarize(X2)

    X3 = np.zeros((N, 32, 22, 22), dtype=np.float32)
    for b in range(N):
        for co in range(32):
            acc = np.zeros((22, 22), dtype=np.float32)
            for ci in range(32):
                kernel = conv3w[co, ci][::-1, ::-1]
                acc += convolve2d(X2[b, ci], kernel, mode="valid")
            X3[b, co] = acc
    X3 = binarize(X3)

    P3 = avg_pool2d(X3, kernel=2, stride=2)    
    F3 = P3.reshape(N, -1)                    
    F3_b = binarize(F3)                 
   
    logits_no_bias = F3_b.dot(fc_w.T)  # (N,10) binary, no bias
    return logits_no_bias

batch_size = 100
correct_no_bias = 0
for i in trange((len(X) + batch_size - 1)//batch_size):
    xb = X[i*batch_size:(i+1)*batch_size]
    yb = y[i*batch_size:(i+1)*batch_size]
    out4 = feed_forward(xb)
    preds_no_bias = out4.argmax(axis=1)
    correct_no_bias += np.sum(preds_no_bias == yb)

acc_no_bias = correct_no_bias / len(X) * 100
print(f"Final Accuracy: {acc_no_bias:.2f}%")

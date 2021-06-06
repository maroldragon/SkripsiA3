#!/usr/bin/env python

import numpy as np
import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import codecs
import pandas as pd
import argparse
import pickle
import random
import sys

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--alpha', type=float, default="0.02",
                    help='The value of alpha')
parser.add_argument('--beta', type=float, default='0.02',
                    help='The beta of beta')
parser.add_argument('--lambda_val', type=float, default='0.02',
                    help='The value of lambda')
parser.add_argument('--corrupt_ratio', type=float, default='0.004',
                    help='The ratio for mSDA')
args = parser.parse_args()

############################
# Define parameters
############################
random.seed(1)
training_ratio = 0.5
num_item = 100
[m, p] = [5,5]
[n, q] = [5,5]

d = 5

R = pd.DataFrame(
[[2.0,4.0,1.0,0.0,0],
[0.0,0,	0.0,1.0,2.0],
[5.0,1.0,1.0,0.0,0],
[0,0.0,	3.0,7.0,0.0],
[0.0,0,	0.0,0.4,0.0]])


X = pd.DataFrame([
[1,0,0,0,1],
[0,1,1,0,0],
[0,0,0,1,0],
[0,1,1,0,1],
[1,0,0,1,0]])


Y = pd.DataFrame([
[1,0,0,0,0],
[0,0,1,0,0],
[0,1,0,0,0],
[0,0,0,0,1],
[0,0,0,1,0]])


X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
 
mask_detail_train = pd.DataFrame([
[1,1,1,0,0],
[0,0,0,1,1],
[1,1,1,0,0],
[0,0,1,1,0],
[0,0,0,1,0]])

A = np.asarray(mask_detail_train)

##normalization
R_mean = np.mean(np.asarray(R)[np.where(A == 1)])
R_std = np.std(np.asarray(R)[np.where(A == 1)])
R = (R - R_mean) / R_std
R = np.asarray(R, dtype=np.float32)

alpha = args.alpha
beta = args.beta
lambda_val = args.lambda_val
corrupt_ratio = args.corrupt_ratio  # the ratio for mSDA
Epoch = args.Epoch

np.random.seed(100)
W1 = np.random.rand(p, p).astype(np.float32)
P1 = np.random.rand(p, d).astype(np.float32)
W2 = np.random.rand(q, q).astype(np.float32)
P2 = np.random.rand(q, d).astype(np.float32)

############################
# Update rules
############################

def update_P1(W_1, X, U):
    U = U.data
    a = np.dot(np.transpose(U), U)
    b = np.dot(np.dot(W1, X), U)
    a = np.transpose(a)
    b = np.transpose(b)
    return np.transpose(np.linalg.solve(a, b)).astype(np.float32)

def update_P2(W2, Y, V):
    V = V.data
    a = np.dot(np.transpose(V), V)
    b = np.dot(np.dot(W2, Y), V)
    a = np.transpose(a)
    b = np.transpose(b)
    return np.transpose(np.linalg.solve(a, b)).astype(np.float32)

def update_W1(X, lambda_val, corrupt_ratio, P1, U, p):
    U = U.data
    S1 = (1 - corrupt_ratio) * np.dot(X, np.transpose(X))
    S1 += lambda_val * np.dot(P1, np.dot(np.transpose(U), np.transpose(X)))
    Q1 = (1 - corrupt_ratio) * np.dot(X, np.transpose(X))
    tmp = (1 - corrupt_ratio) * (1 - corrupt_ratio) * (np.ones([p, p]) - np.diag(np.ones([p]))) * np.dot(X, np.transpose(X))
    tmp += (1 - corrupt_ratio) * np.diag(np.ones([p])) * np.dot(X, np.transpose(X))
    Q1 += lambda_val * tmp
    return np.linalg.solve(Q1, S1).astype(np.float32)

def update_W2(Y, lambda_val, corrupt_ratio, P2, V, q):
    V = V.data
    S2 = (1 - corrupt_ratio) * np.dot(Y, np.transpose(Y))
    S2 += lambda_val * np.dot(P2, np.dot(np.transpose(V), np.transpose(Y)))
    Q2 = (1 - corrupt_ratio) * np.dot(Y, np.transpose(Y))
    tmp = (1 - corrupt_ratio) * (1 - corrupt_ratio) * (np.ones([q, q]) - np.diag(np.ones([q]))) * np.dot(Y, np.transpose(Y))
    tmp += (1 - corrupt_ratio) * np.diag(np.ones([q])) * np.dot(Y, np.transpose(Y))
    Q2 += lambda_val * tmp
    return np.linalg.solve(Q2, S2).astype(np.float32)

class Model(chainer.Chain):
    def __init__(self, m, n, d):
        super(Model, self).__init__()
        with self.init_scope():
            self.u = L.Linear(d, m)
            self.v = L.Linear(d, n)
            
    def obtain_value(self, ):
        u = self.u.W.data
        v = self.v.W.data
        return [u, v]
    
    def obtain_loss(self, lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta):
        loss = 0
        loss += lambda_val * F.sum(F.square(F.matmul(P1, F.transpose(self.u.W)) - F.matmul(W1, X)))
        loss += lambda_val * F.sum(F.square(F.matmul(P2, F.transpose(self.v.W)) - F.matmul(W2, Y)))
        loss += alpha * F.sum(F.square(A * (R - F.matmul(self.u.W, F.transpose(self.v.W)))))
        loss += beta * ((F.sum(F.square(self.u.W)) + F.sum(F.square(self.v.W))))
        return loss


# model definition
model = Model(m, n, d)
optimizer = optimizers.SGD(0.002)
optimizer.setup(model)
U = model.u.W.data
V = model.v.W.data

loss_temp = sys.maxsize

loss = model.obtain_loss(lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta)
model.zerograds()
loss.backward()
optimizer.update()
U = model.u.W.data
V = model.v.W.data

index = 2

while(loss.data < loss_temp):
    loss_temp = loss.data
    W1 = update_W1(X, lambda_val, corrupt_ratio, P1, U, p) #np.asarray(w1).astype(np.float32)
    W2 = update_W2(Y, lambda_val, corrupt_ratio, P2, V, q) #np.asarray(w2).astype(np.float32)
    P1 = update_P1(W1, X, U) #np.asarray(p1).astype(np.float32)
    P2 = update_P2(W2, Y, V) #np.asarray(p2).astype(np.float32)
    model.zerograds()
    loss = model.obtain_loss(lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta)

    loss.backward()
    optimizer.update()
    U = model.u.W.data
    V = model.v.W.data

##    print("##############################", index)
##    print("U", index)
##    for i in range(len(U[0])):
##        print(i+1)
##        for j in range(len(U)):
##            print((U[j][i]))
##        print()
##    print("===============================")
##    print("V", index)
##    for i in range(len(V[0])):
##        print(i+1)
##        for j in range(len(V)):
##            print((V[j][i]))
##        print()
##    index += 1
##for epoch in range(Epoch):
##    loss = model.obtain_loss(lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta)
##    
##    model.zerograds()
####    loss = model.obtain_loss(lambda_val, P1, W1, X, P2, W2, Y, A, R, alpha, beta)
##    print("epoch", loss)
##    
##    loss.backward()
##    optimizer.update()
##    U = model.u.W.data
##    V = model.v.W.data
##    
####    print("V")
####    for i in range(len(V[0])):
####        print(i+1)
####        for j in range(len(V)):
####            print((V[j][i]))
####        print()
    
output = [A,R,X,Y,W1,W2,P1,P2]

numU = np.array(U.data)
numV = np.array(V.data)

##print("R")
result = np.matmul(numU,np.transpose(numV))
##print(result)
print(L.Linear(d, m))

with open('output.dump', 'wb') as f:
    pickle.dump(output, f)

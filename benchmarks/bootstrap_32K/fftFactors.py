# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np

class DiagonalMatrix:
    _n_: int
    _entries_: np.ndarray
    _index_: int

    def __init__(self, entries, index):
        self._entries_ = entries
        self._n_ = len(entries)
        self._index_ = index 

    def __add__(self, other):
        if self._n_ != other._n_:
            raise Exception("Diagonal Matricies must have the same dimensions")
        if self._index_ != other._index_:
            raise Exception("Diagonal Matricies must have the same index")
        new_entries = np.zeros((self._n_), dtype=complex)
        for i in range(self._n_):
            new_entries[i] = self._entries_[i] + other._entries_[i]
        return DiagonalMatrix(new_entries,self._index_)

    def index(self):
        return self._index_

    def entries(self):
        return self._entries_
    
    def __mul__(self, other):
        if not isinstance(other, DiagonalMatrix):
            new_entries = self._entries_ * other
            return DiagonalMatrix(new_entries,self._index_)
        if self._n_ != other._n_:
            raise Exception("Diagonal Matricies must have the same dimensions")
        # new_entries = [0]*self._n_
        new_entries = np.zeros((self._n_), dtype=complex)
        _self_rotated_ = self._entries_
        _other_rotated_ = np.concatenate((other._entries_[self._index_:] , other._entries_[:self._index_]))

        new_index = self._index_ + other._index_
        if new_index >= self._n_//2:
            new_index = new_index - self._n_
        elif new_index < -self._n_//2:
            new_index = self._n_ + new_index

        for i in range(self._n_):
            new_entries[i] = _self_rotated_[i] * _other_rotated_[i]
        return DiagonalMatrix(new_entries,new_index)

    def to_matrix(self):
        mat = np.zeros([self._n_,self._n_], dtype=complex)
        for i in range(self._n_):
            j = (i + self._index_) % self._n_
            mat[i][j] = self._entries_[i]
        return mat

    def __repr__(self):
        return "DiagonalMatrix{" + str(self._entries_) + " : " + str(self._index_) + "}"

def computeRoots(N):
    m = N << 1
    roots = [0]*m
    roots[0] = 1
    for i in range(1,m):
        angle = 2*math.pi*i/m
        roots[i] = math.cos(angle) + 1j * math.sin(angle)
    return roots

def computePow5(N):
    pows = [0]*(N+1)
    pows[0] = 1
    for i in range(1,N+1):
        pows[i] = (pows[i-1]*5) & ((N << 1) - 1)
    return pows

def fftInvPlainVec(slots):
    roots = computeRoots(slots << 1)
    pow5 = computePow5(slots << 1)
    
    factors = []

    N = slots
    m = N
    while m >= 2:
        tt = m >> 1
        gap = N // m
        mask = (m << 2) - 1
        a = np.zeros((slots), dtype=complex)
        b = np.zeros((slots), dtype=complex)
        c = np.zeros((slots), dtype=complex)
        for i in range(0,N,m):
            for j in range(0,tt):
                idx1 = i + j
                idx2 = i + j + tt
                k = ((m << 2) - (pow5[j] & mask))* gap
                if m == N:
                    a[idx1] = 1/2
                    a[idx2] = -roots[k]/2
                    c[idx1] = 1/2
                    c[idx2] = roots[k]/2
                else:
                    a[idx1] = 1/2
                    a[idx2] = -roots[k]/2
                    b[idx1] = 1/2
                    c[idx2] = roots[k]/2
        a = DiagonalMatrix(a,0)
        b = DiagonalMatrix(b,tt)
        c = DiagonalMatrix(c,-tt)
        if m == N:
            factors.append([a,c])
        else: 
            factors.append([a,b,c])
        m = m >> 1
    return factors

def fftPlainVec(slots):
    roots = computeRoots(slots << 1)
    pow5 = computePow5(slots << 1)

    factors = []
    N = slots
    m = 2
    while m <= N:
        tt = m >> 1
        gap = N // m
        mask = (m << 2) - 1
        a = np.zeros((slots), dtype=complex)
        b = np.zeros((slots), dtype=complex)
        c = np.zeros((slots), dtype=complex)

        for i in range(0,N,m):
            for j in range(0,tt):
                idx1 = i + j
                idx2 = i + j + tt
                k = (pow5[j] & mask)* gap
                if m == N:
                    a[idx1] = 1
                    a[idx2] = -roots[k]
                    c[idx1] = roots[k]
                    c[idx2] = 1
                else:
                    a[idx1] = 1
                    a[idx2] = -roots[k]
                    b[idx1] = roots[k]
                    c[idx2] = 1
        a = DiagonalMatrix(a,0)
        b = DiagonalMatrix(b,tt)
        c = DiagonalMatrix(c,-tt)
        if m == N:
            factors.append([a,c])
        else: 
            factors.append([a,b,c])
        m = m << 1
    return factors

def generate_matrix(diags):
    _n_ = diags[0]._n_
    mat = np.zeros([_n_,_n_],dtype=complex)
    mask = _n_ 
    for m in diags:
        for i in range(_n_):
            j = (i + m._index_) % _n_
            mat[i][j] = m._entries_[i]
    return mat

def merge_levels(slots,factors,depths,inverse):
    depth_count = 0
    cnt = 0
    i = 0
    merged = []
    while cnt < len(factors):
        depth = depths[i]
        fs = {}
        for f in factors[cnt]:
            if f.index() in fs.keys():
                fs[f.index()] = fs[f.index()] + f
            else:
                fs[f.index()] = f
        cnt = cnt + 1
        i = i + 1
        for j in range(1,depth):
            fs_copy = fs.copy()
            fs = {}
            for f1 in factors[cnt]:
                for f2 in fs_copy.values():
                    mul = f1 * f2
                    if mul.index() in fs.keys():
                        fs[mul.index()] = fs[mul.index()] + mul
                    else:
                        fs[mul.index()] = mul
            cnt = cnt + 1
        if inverse:
            reshape_shape = slots // (1 << (depth_count))
            depth_count += depth
        else:
            depth_count += depth
            reshape_shape = (1 << (depth_count))
        for k in fs.keys():
            fs[k]._entries_ = fs[k]._entries_[0:reshape_shape]
        merged.append(fs)
    return merged 


def genFftMatrices(slots,maxDepth):
    depths = []
    logSlots = int(math.log2(slots))
    dpth = 0
    while dpth < logSlots:
        depths.append(min(maxDepth, logSlots - dpth))
        dpth = dpth + depths[-1]

    depths.reverse()
    factors = fftPlainVec(slots)
    fftMatrices = merge_levels(slots,factors,depths,inverse=False)
    return fftMatrices

def genInvFftMatrices(slots,maxDepth):
    depths = []
    dpth = 0
    logSlots = int(math.log2(slots))
    while dpth < logSlots:
        depths.append(min(maxDepth, logSlots - dpth))
        dpth = dpth + depths[-1]

    factors = fftInvPlainVec(slots)
    invFftMatrices = merge_levels(slots,factors,depths,inverse=True)
    return invFftMatrices
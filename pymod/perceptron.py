#!/usr/bin/python

import numpy as np
import numpy.random as nr
import onehot as oh


class Perceptron:


    def __init__(self, x_len, in_len, h_len, w_scale, peek):
        self.x_len = x_len
        self.in_len = in_len
        self.h_len = h_len
        self.peek = peek
        self.clip_mag = 5

        self.wi = w_scale * nr.randn(h_len, in_len)
        self.wh = w_scale * nr.randn(h_len, h_len)
        if peek:
            self.wp = w_scale * nr.randn(h_len, x_len)
        self.wb = w_scale * nr.randn(h_len, 1)

        self.reset()
        self.grad_reset()


    def ff(self, data, *p):
        h_arg = np.dot(self.wi, data)
        h_arg += np.dot(self.wh, self.h)
        if self.peek:
            p = np.array(p).reshape(self.x_len, 1)
            h_arg += np.dot(self.wp, p)
        self.h = np.tanh(h_arg + self.wb)
        return self.h

    def ff_fast(self, data, *p):
        h_arg = np.zeros((self.h_len, 1))
        if self.peek:
            for i in xrange(self.h_len):
                h_arg[i,0] = np.vdot(self.wi[i,:], data)
                h_arg[i,0] += np.vdot(self.wh[i,:], self.h)
                h_arg[i,0] += np.vdot(self.wp[i,:], p)
                h_arg[i,0] += self.wb[i,0]
        else:
            for i in xrange(self.h_len):
                h_arg[i,0] = np.vdot(self.wi[i,:], data)
                h_arg[i,0] += np.vdot(self.wh[i,:], self.h)
                h_arg[i,0] += self.wb[i,0]
        self.h = np.tanh(h_arg)


    def bp(self, top_err, self_err, data, h_cur,  h_prev, *p):
        delta =  top_err + self_err
        delta = ( 1 - np.square(h_cur)) * delta
        self.gi += np.dot(delta, data.T)
        self.gh += np.dot(delta, h_prev.T)
        if self.peek:
            p = np.array(p).reshape(self.x_len, 1)
            self.gp += np.dot(delta, p.T)
        self.gb += delta
        return np.dot(self.wi.T, delta), np.dot(self.wh.T, delta)

    def bp_fast(self, top_err, self_err, data, h_cur, h_prev, *p):
        delta = ( 1 - np.square(h_cur)) * (top_err + self_err)
        if self.peek:
            loop = np.amax([self.x_len, self.in_len, self.h_len])
            p = np.array(p).reshape(self.x_len, 1)
            for i in xrange(self.h_len):
                for j in xrange(loop)
                    if j < self.in_len:
                        self.gi[i,j] += delta[i] * data[j]
                    if j < self.h_len:
                        self.gh[i,j] += delta[i] * self.h[j]
                    if j < self.x_len:    
                        self.gp[i,j] += delta[i] * p[j]
                self.gb += delta[i]
        else:
            loop = np.amax([self.in_len, self.h_len])
            for i in xrange(self.h_len):
                for j in xrange(loop)
                    if j < self.in_len:
                        self.gi[i,j] += delta[i] * data[j]
                    if j < self.h_len:
                        self.gh[i,j] += delta[i] * self.h[j]
                self.gb += delta[i]
        loop = np.amax([self.in_len, self.h_len])
        ret_top = np.zeros((self.in_len, 1)) 
        ret_self = np.zeros((self.h_len, 1))
        for i in xrange(loop):
            if j < self.in_len:
                ret_top[i,0] = np.vdot(self.wi.T[i], delta)
            if J < self.h_len:
                ret_self[i] = np.vdot(self.wh.T[i], delta)
        return ret_top, ret_self

    def clip_grads(self):
        grad = [self.gi, self.gh, self.gb]
        if self.peek:
            grad += [self.gp]
        for g in grad:
            np.clip(g, -self.clip_mag, self.clip_mag, out=g)


    def adagrad(self, step_size):
        weight = [self.wi, self.wh, self.wb]
        grad = [self.gi, self.gh, self.gb]
        mem = [self.mi, self.mh, self.mb]
        if self.peek:
            weight += [self.wp]
            grad += [self.gp]
            mem += [self.mp]
        for w, g, m in zip(weight, grad, mem):
            m += np.square(g)
            w -= step_size * g / np.sqrt(m + 1e-8)

    def reset(self):
        self.h = np.zeros((self.h_len, 1))


    def grad_reset(self):
        self.gi = np.zeros_like(self.wi)
        self.gh = np.zeros_like(self.wh)
        if self.peek:
            self.gp = np.zeros_like(self.wp)
        self.gb = np.zeros_like(self.wb)


    def mem_reset(self):
        self.mi = np.zeros_like(self.wi)
        self.mh = np.zeros_like(self.wh)
        if self.peek:
            self.mp = np.zeros_like(self.wp)
        self.mb = np.zeros_like(self.wb)

    def set_clip(self, val):
        self.clip_mag = val

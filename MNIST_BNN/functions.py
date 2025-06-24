# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:30:49 2018

@author: akash

This code is almost similar to the one in https://github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch.git.
We changed the code of BinarizeF to avoid floating point operation which shows -0.0 and 0.0 in the output.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input > 0] = 1
        output[input <= 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply
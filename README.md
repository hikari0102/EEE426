# MNIST Classification using Binarized Neural Networks (BNN)

Final project for **EEE426 - Tensor Processor Design for Image Recognition**, UNIST.

This project implements a Binarized Neural Network (BNN) for MNIST digit classification, trained in PyTorch and accelerated on an FPGA (PYNQ-Z2). The design achieves high accuracy and efficient inference performance exceeding known empirical limits for raw streaming convolution.

---

## Project Highlights

- **Accuracy**: 96.55% on MNIST test set  
- **Inference Time**: 25.9 seconds on PYNQ-Z2 board  
  → Reachs the empirical upper bound of 25 seconds for stream-based raw convolution  
- **Platform**: [PYNQ-Z2](http://www.pynq.io/board.html)

---

## Folder Structure
'''
├── HLScodes/
│ ├── High-Level Synthesis (HLS) codes used for hardware acceleration on PYNQ-Z2
│ └── Testbenches for verifying functionality
│
├── MNIST_BNN/
│ ├── Python scripts for training and inference of BNN on MNIST dataset
│ └── Based on: https://github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch
│
└── final_report - Final project report with top-level diagram and design motivations
'''
---

## Reference

- [Akashmathwani/Binarized-Neural-networks-using-pytorch](https://github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch)

---

## Authors

This repository was developed as part of the final coursework for **EEE426 @ UNIST**.  
Tensor Processor Design for Image Recognition, Spring Semester.

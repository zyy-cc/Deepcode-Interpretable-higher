# Deepcode-Interpretability-Higher

## Overview
This repository contains the codes for the paper "Higher-order Interpretations of DeepCode, a Learned Feedback Codes". It is a non-linear interpretable model with higher-order error correction for AWGN channel with feedback.

## Structure

- **deepcode.py**  
  Implementation of Deepcode using PyTorch, based on TensorFlow Deepcode.
  
- **encx_deepdec5.py**  
  Ablation study on different encoders with a Deepcode decoder.
  
- **enc3decxsingle.py**  
  Encoder with third-order error correction.  
  Decoder that considers future x-1 parity bits.  
  Single-stage decoder.
  
- **enc3decxtwofix.py**  
  Encoder with third-order error correction.  
  Decoder that considers future x-1 parity bits.  
  Two-stage decoder with fixed knee points (noiseless feedback).
  
- **enc3decxtwovary.py**  
  Encoder with third-order error correction.  
  Decoder that considers future x-1 parity bits.  
  Two-stage decoder with varying knee points (noisy feedback).


The `logs` folder contains the parameters for all models.

original TensorFlow Deepcode: https://github.com/hyejikim1/Deepcode

previous interpretable model: https://github.com/zyy-cc/Deepcode-Interpretability


# Tutorial: Bloch Model-Based Reconstruction in BART

This tutorial provides guidance of how to run the Bloch model-based reconstruction with the BART toolbox. It presents examples for an IR bSSFP and IR FLASH reconstruction. In both cases a simulated digital reference object is created and the parameter maps for $T_1$, $T_2$ and $FA_{\text{rel}}$ are compared to references.

The theoretical background of this work can be found in our manuscript titled

	Quantitative Magnetic Resonance Imaging by Nonlinear Inversion of the Bloch Equations

by Nick Scholand, Xiaoqing Wang, Volkert Roeloffs, Sebastian Rosenzweig, Martin Uecker.

The examples are interactive. You can change any part of the code in the provided jupyter-notebook. We recommend to run it on Google Colab (follow link below), because it provides all required hardware and should run out-of-the-box. If you want to run it locally, please skip the "0. Setup BART on Google Colab" chapter and have the latest BART version installed.

**Bloch Model-Based Reconstruction Tutorial**
- [Jupyter Notebook](./bart-bloch-tutorial.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/bloch-tutorial/blob/master/bart-bloch-tutorial.ipynb)
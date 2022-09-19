---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="DDooP3kQ2N6w" -->
# Tutorial: Bloch Model-Based Reconstruction in BART


    Author:        Nick Scholand
    Email:         scholand@tugraz.at
    Institution:   Graz University of Technology, Graz, Austria

**About the Tutorial**

This tutorial introduces the Bloch model-based reconstruction tool added with the `--bloch` flag to the `moba` tool in the [official BART repository](https://github.com/mrirecon/bart).
The `--bloch` option in `moba` can run on the CPU only, but it is highly recommended to have a GPU. Thus, this example is optimized for the Google Colab service providing the hardware infrastructure including a GPU.
<!-- #endregion -->

<!-- #region id="lZdzii_U2N60" -->
### 0. Setup BART on Google Colab

The following section sets up BART on Google Colab. For a detailed explanation, see [**How to Run BART on Google Colaboratory**](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2021).

Please skip this part and click [here](#main) if you want to run this notebook on your local machine with BART already installed.

If you want to use a GPU instance, please turn it on in Google Colab:

    Go to Edit → Notebook Settings
    Choose GPU from Hardware Accelerator drop-down menu

<!-- #endregion -->

<!-- #region id="2hjTBx3ZJcwE" -->
We check which GPU instance was assigned to this notebook.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="5_gw1eAr2N62" outputId="e2b3273e-f5c3-4633-9c65-0202cf584dc2"

# Use CUDA 10.1 when on Tesla K80

# Estimate GPU Type
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

echo "GPU Type:"
echo $GPU_NAME

if [ "Tesla K80" = "$GPU_NAME" ];
then
    echo "GPU type Tesla K80 does not support CUDA 11. Set CUDA to version 10.1."

    # Change default CUDA to version 10.1
    cd /usr/local
    rm cuda
    ln -s cuda-10.1 cuda
else
    echo "Current GPU supports default CUDA-11."
    echo "No further actions are required."
fi

echo "GPU Information:"
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
nvcc --version
```

<!-- #region id="RrQOgLBbJndV" -->
We download the current master branch version of BART and install its dependencies.
<!-- #endregion -->

```bash id="WuC1x6fs2N64"

# Install BARTs dependencies
apt-get install -y make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev bc &> /dev/null

# Clone Bart
[ -d /content/bart ] && rm -r /content/bart
git clone https://github.com/mrirecon/bart/ bart &> /dev/null
```

<!-- #region id="TRmUblbXJ0Gy" -->
We compile BART with some settings for running it on the GPU on Google Colab.
<!-- #endregion -->

```bash id="itZOCWs52N66"

BRANCH=master

cd bart

# Switch to desired branch of the BART project
git checkout -q $BRANCH

# Define specifications 
COMPILE_SPECS=" PARALLEL=1
                CUDA=1
                CUDA_BASE=/usr/local/cuda
                CUDA_LIB=lib64
                OPENBLAS=1
                BLAS_THREADSAFE=1"

printf "%s\n" $COMPILE_SPECS > Makefiles/Makefile.local

make &> /dev/null
```

<!-- #region id="4mb80KgrJ-R4" -->
We set the required environment variables for BART.
<!-- #endregion -->

```python id="QxYZ5xp_2N68"
import os
import sys

# Define environment variables for BART and OpenMP

os.environ['TOOLBOX_PATH'] = "/content/bart"

os.environ['OMP_NUM_THREADS']="4"

# Add the BARTs toolbox to the PATH variable

os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python")
```

<!-- #region id="bSvHAahc2N69" -->
We check the installed BART version.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="HIQs5a1S2N69" outputId="2ee36e5f-82b6-47d8-9d42-891212f0fbeb"

echo "# The BART used in this notebook:"
which bart
echo "# BART version: "
bart version
```

<!-- #region id="yofneb2U2N6_" -->
<a name='main'></a>
### Functions for Visualization

We define a function for visualization of the results.
<!-- #endregion -->

```python id="HGKRhAhq2N7A"
import sys
import os
sys.path.insert(0, os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
import cfl

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
def diffplot(reco, ref, vmax=3, cmap='viridis', title="Nothing"):
    
    DIFF_SCALING = 20
    
    reco = np.abs(cfl.readcfl(reco).squeeze())
    ref = np.abs(cfl.readcfl(ref).squeeze())
    
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    
    # Reference
    
    im = ax1.imshow(ref, cmap=cmap, vmin=0, vmax=vmax)
    
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    cax.set_visible(False)
    
    ax1.set_title('Reference')
    
    # Reconstruction
    
    ax2 = fig.add_subplot(132)

    im2 = ax2.imshow(reco, cmap=cmap, vmin=0, vmax=vmax)
    
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label(title, fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    cax.set_visible(False)
    
    ax2.set_title('Reconstruction')
    
    # Difference
    
    ax3 = fig.add_subplot(133)

    im3 = ax3.imshow(np.abs(reco-ref)*DIFF_SCALING, cmap=cmap, vmin=0, vmax=vmax)
    
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im3, cax=cax)
    cbar.set_label(title, fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    
    ax3.set_title('Difference')
    ax3.text(0.01*np.shape(reco)[0], 0.95*np.shape(reco)[0], 'x'+str(DIFF_SCALING), fontsize=20, color='white')

```

<!-- #region id="XohWaTap2N7D" -->
### Previous Tutorials

This tutorial assumes basic knowledge about BART cmdline tools. Please check out the following tutorials:
- Introduction to BART Cmdline tools: [Link to Tutorial](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2019)
- Simulation tool in BART: [Link to Tutorial](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2022)

<!-- #endregion -->

<!-- #region id="hpkprAXRocoG" -->
## IR bSSFP Reconstruction

In the first example the Bloch model-based reconstruction is used to determine quantitative parameter maps from an IR bSSFP sequence.
<!-- #endregion -->

<!-- #region id="tvLLFZyi2N7E" -->
### Create Dataset

The numerical dataset is created using BARTs `sim` and `phantom` tool.
Here, for its simplicity a single tube is simulated in k-space using a golden-angle sampled single-shot IR bSSFP sequence.

More complex geometries like the NIST phantom require a higher resolution to resolve the relatively smaller individual tubes. Thus, they are too computationally demanding for this simple tutorial.
<!-- #endregion -->

```bash id="xg8Hi4N92N7E" outputId="ed293858-5c74-4f13-8736-fd30997b8f09" colab={"base_uri": "https://localhost:8080/"}

# Sequence Parameter
TR=0.0045 #[s]
TE=0.00225 #[s]
TRF=0.001 #[s]
REP=600
FA=45 #[deg]
BWTP=4

# Tissue Parameter
T1=0.8 # [s]
T2=0.1 # [s]

# Run simulation and save output to $DURATION variable defining the simulation time
bart sim --ODE --seq IR-BSSFP,TR=$TR,TE=$TE,Nrep=$REP,pinv,ppl=$TE,Trf=$TRF,FA=$FA,BWTP=$BWTP -1 $T1:$T1:1 -2 $T2:$T2:1 simu

# Create Golden-Angle based Trajectory with 2-fold oversampling
SAMPLES=30
SPOKES=1

bart traj -x $((2*SAMPLES)) -y $SPOKES -o 2 -G -t $REP _traj

## The moba tool requires the time dimension to be in (5)!
bart transpose 5 10 _traj traj


# Simulate Spatial Components

## Create phantom based on trajectory
bart phantom -c -k -t traj comp_geom_ksp

# Combine simulated signal and spatial basis to create numerical dataset
bart fmac -s $(bart bitmask 6) comp_geom_ksp simu phantom_ksp
```

<!-- #region id="mi7vUtW92N7G" -->
**Inversion Time**  

The inversion time is historically used by the `moba` tool to pass the time information to the underlying analytical forward operators. The Bloch model-based tool, does not require this vector. The timing information is included in the description of the simulated sequence: in the repetition time, the echo time, the number of repetitions,...

Still for compatibility to the previous versions the inversion time file TI still needs to be bassed. This will become optional in future releases of BART. For now, we create a dummy inversion time file.
<!-- #endregion -->

```bash id="HfPUkVVN2N7G"

# Create dummy inversion time file

REP=600
bart index 5 ${REP} TI
```

<!-- #region id="2r3mH7Lb2N7H" -->
### Reconstruction

For allow the `moba` tool to use the Bloch model operator `--bloch` is added.

The information about the applied sequence is passed with the `--seq` interface.
More details are requested with:
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="s98psZ-52N7H" outputId="077379cb-d65c-4c96-8567-39f1531fb497"

bart moba --seq h
```

<!-- #region id="3Z_7G_Et2N7J" -->
In this example, we pass the sequence type, the repetition and echo time and many more parameter:
`ir-bssfp,tr=${TR},te=${TE},...`

The simulation type can be controlled with the `moba --sim` interface:
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="FYnPIV8_2N7J" outputId="dd7029e4-d32b-4265-fdb8-ff547c8ea0e2"

bart moba --sim h
```

<!-- #region id="50FyP5OF2N7K" -->
Currently, it supports an ordinary differential equation solver (ODE) and a state-transition matrix (STM) simulation of the Bloch equations.

The flag `--img_dims $DIM:$DIM:1` defines the $k_{\text{max}}$ of the actual k-space by taking the two-fold oversampling into account.

The `moba --other` interface allows for passing further parameters like the partial derivative scalings, which define the preconditioning of the reconstructed parameter maps: [$R_1$, $M_0$, $R_2$, $B_1$]. With `--other pdscale=1:1:3:0` the optimization of the $B_1$ mapping is turned off (set to 0), the algorithm optimizes for $\hat{R_2}=3\cdot R_2$ and the other parameter maps are untouched.
In praxis this preconditioning often need to be tuned for a smooth convergence of the optimization. The scaling depends on the sequence type and its individual parameters. Later we will show an additional example for an IR FLASH sequence.

Besides tuning the preconditioning, we need to set the initialization values for the reconstruction `pinit=3:1:1:1`. Both are part of moba's `--other` interface. Please run `bart moba --other h` for more details. Further, the reconstruction needs some information about data and PSF scaling: `--scale_data=5000. --scale_psf=1000. --normalize_scaling`.

The iteratively regularized Gauss-Newton method and FISTA are both controlled similar to previous model-based publications by Xiaoqing Wang:

* Wang, X., Roeloffs, V., Klosowski, J., Tan, Z., Voit, D., Uecker, M. and Frahm, J. (2018), Model-based T1 mapping with sparsity constraints using single-shot inversion-recovery radial FLASH. Magn. Reson. Med, 79: 730-740. https://doi.org/10.1002/mrm.26726

*  Wang, X, Rosenzweig, S, Scholand, N, Holme, HCM, Uecker, M. Model-based reconstruction for simultaneous multi-slice T1 mapping using single-shot inversion-recovery radial FLASH. Magn Reson Med. 2020; 85: 1258– 1271. https://doi.org/10.1002/mrm.28497

* Wang, X, Tan, Z, Scholand, N, Roeloffs, V, Uecker, M. Physics-based reconstruction methods for magnetic resonance imaging. Phil Trans R Soc. 2021; A379: 20200196. https://doi.org/10.1098/rsta.2020.0196   .

we will not go into detail here, but the passed parameters are: `-i$ITER -C$INNER_ITER -s$STEP_SIZE -B$MIN_R1 -o$OS -R$REDU_FAC -j$LAMBDA -N`.

After discussing the interface, the next cell will execute the reconstruction. Depending on your system and the GPU this step can take about a minute.

If you do not want to use a GPU, please remove the `-g` flag.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="DH1sUBHs2N7L" outputId="450a9b75-0f4e-49b5-8ea1-d4029a546b51"

OS=1
REDU_FAC=3
INNER_ITER=250
STEP_SIZE=0.95
MIN_R1=0.001
ITER=8
LAMBDA=0.0005

DIM=30
TR=0.0045 #[s]
TE=0.00225 #[s]
TRF=0.001 #[s]
REP=600
FA=45 #[deg]
BWTP=4

bart moba --bloch --sim STM --img_dims $DIM:$DIM:1 \
        --seq IR-BSSFP,TR=${TR},TE=${TE},ppl=${TE},FA=${FA},Trf=${TRF},BWTP=${BWTP},pinv \
        --other pinit=3:1:1:1,pscale=1:1:3:0 --scale_data=5000. --scale_psf=1000. --normalize_scaling \
        -g \
        -i$ITER -C$INNER_ITER -s$STEP_SIZE -B$MIN_R1 -d 4 -o$OS -R$REDU_FAC -j$LAMBDA -N \
        -t traj phantom_ksp TI reco sens
```

<!-- #region id="qaWh_vYk2N7M" -->
The chosen debug level of `-d 4` leads to the output above. It is highly recommended for direct feedback from the reconstruction.
<!-- #endregion -->

<!-- #region id="G6sw6nrM2N7M" -->
### Post-Processing

The output of the Bloch model-based reconstruction has the dimensions
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="mDvjMyq42N7N" outputId="79e4d7ab-f184-423c-d4a1-a8481140f603"

cat reco.hdr
```

<!-- #region id="KYcsmtaI2N7O" -->
60 x 60 samples for each of the 4 parameter maps.

Currently, the two-fold oversampling is not automatically compensated. Thus, we resize the data with `bart resize -c 0 $DIM 1 $DIM ...`.

The 6th dimension includes the parameter maps $R_1$, $M_0$, $R_2$ and $B_1$. We slice them `bart slice 6 ...` and invert the relaxation maps `bart spow -- -1 ...` to get the actual parameter maps $T_1$, $T_2$ and $B_1$. Multiplying the maps with a mask `bart fmac ...` can improve the visualization.
<!-- #endregion -->

```bash id="cew_vvrW2N7O"

# Post-Process Reconstruction

DIM=30
T1=0.8 #[s]
T2=0.1 #[s]

# Create Reference Maps

bart phantom -c -x $DIM circ

bart scale -- $T1 circ t1ref
bart scale -- $T2 circ t2ref
bart copy circ faref


# Resize output of Reconstruction
# -> compensate for 2-fold oversampling
bart resize -c 0 $DIM 1 $DIM reco reco_crop


# Convert and Mask Reconstructed Maps

bart slice 6 0 reco_crop r1map
bart spow -- -1 r1map _t1map
bart fmac _t1map circ t1map #mask
rm _t1map.cfl _t1map.hdr

bart slice 6 2 reco_crop r2map
bart spow -- -1 r2map _t2map
bart fmac _t2map circ t2map #mask
rm _t2map.cfl _t2map.hdr

bart slice 6 3 reco_crop _famap
bart fmac _famap circ famap #mask
rm _famap.cfl _famap.hdr
```

<!-- #region id="a7EX8E5A2N7P" -->
Finally, the maps can be visualized.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 404} id="a9N_K7932N7P" outputId="e122774e-0a3d-4f00-bc56-e8dade06d047"
diffplot('t1map', 't1ref', 2, 'viridis', 'T$_1$ / s')
diffplot('t2map', 't2ref', 0.2, 'copper', 'T$_2$ / s')
diffplot('famap', 'faref', 1.1, 'hot', 'FA/FA$_{nom}$')
```

<!-- #region id="7bvVsrrl2N7Q" -->
The difference in the relative flip angle map is zero, because the flipangle optimization was turned off by setting the preconditioning scaling to 0: `pscale=1:1:3:0`.

We can observe some small checkboard artifacts and differences close to the edges of the maps, but the overall error for the reconstructed $T_1$ and $T_2$ maps is small.
<!-- #endregion -->

<!-- #region id="9Qd8CsYloyZl" -->
## IR FLASH Reconstruction

Here, we present the reconstruction of IR FLASH data with the Bloch model-based reconstruction tool.
<!-- #endregion -->

<!-- #region id="jbu73cVHsGLf" -->
### Create Dataset

The simulated data is created in the same way as presented in the example with an IR bSSFP sequence. Only some sequence parameters like TR, TE and the flipangle have been modified.
<!-- #endregion -->

```bash outputId="b446c98b-7a1b-4502-b5e8-0fa1f8d980b8" colab={"base_uri": "https://localhost:8080/"} id="UefoXLLrovyy"

# Sequence Parameter
TR=0.003 #[s]
TE=0.0017 #[s]
TRF=0.001 #[s]
REP=600
FA=8 #[deg]
BWTP=4

# Tissue Parameter
T1=0.8 # [s]
T2=0.1 # [s]

# Run simulation and save output to $DURATION variable defining the simulation time
bart sim --ODE --seq IR-FLASH,TR=$TR,TE=$TE,Nrep=$REP,pinv,ppl=$TE,Trf=$TRF,FA=$FA,BWTP=$BWTP -1 $T1:$T1:1 -2 $T2:$T2:1 simu

# Create Golden-Angle based Trajectory with 2-fold oversampling
SAMPLES=30
SPOKES=1

bart traj -x $((2*SAMPLES)) -y $SPOKES -G -t $REP _traj

## The moba tool requires the time dimension to be in (5)!
bart transpose 5 10 _traj traj


# Simulate Spatial Components

## Create phantom based on trajectory
bart phantom -c -k -t traj comp_geom_ksp

# Combine simulated signal and spatial basis to create numerical dataset
bart fmac -s $(bart bitmask 6) comp_geom_ksp simu phantom_ksp
```

<!-- #region id="_BgANAdYsK4J" -->
### Reconstruction

The reconstruction works similar as in the IR bSSFP case. Only the modified sequence parameters and the partial scaling need to be adjusted.

In the IR bSSFP reconstruction we used a scaling of `pscale=1:1:3:0`, which turned off the flipangle optimization and preconditioned to optimize for $\hat{R_2}=3\cdot R_2$ instead of $R_2$. An IR FLASH sequence is sensitive to $T_1$, $M_0$ and the flipangle. Therefore, the scaling is set to `pscale=1:1:0:1`. It constraints $R_2$ to be fixed and optimized for the flipangle. To observe an effect of the flipangle optimization the reconstruction assumes a nominal flipangle of 6 degree, which differs from the simulation. Thus, we expect a resulting relative flipangle map of $8/6=1.\bar{33}$.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} outputId="a599afc0-e28a-47eb-ed33-8fe9e15e1a7c" id="gtYgPACYo_8m"

OS=1
REDU_FAC=3
INNER_ITER=250
STEP_SIZE=0.95
MIN_R1=0.001
ITER=8
LAMBDA=0.0005

DIM=30
TR=0.003 #[s]
TE=0.0017 #[s]
TRF=0.001 #[s]
REP=600
FA=6 #[deg] ! Note: not the same as in the simulation !
BWTP=4

bart moba --bloch --sim STM --img_dims $DIM:$DIM:1 \
        --seq IR-FLASH,TR=${TR},TE=${TE},ppl=${TE},FA=${FA},Trf=${TRF},BWTP=${BWTP},pinv \
        --other pinit=3:1:1:1,pscale=1:1:0:1 --scale_data=5000. --scale_psf=1000. --normalize_scaling \
        -g \
        -i$ITER -C$INNER_ITER -s$STEP_SIZE -B$MIN_R1 -d 4 -o$OS -R$REDU_FAC -j$LAMBDA -N \
        -t traj phantom_ksp TI reco sens
```

<!-- #region id="SOE3KXUBsOFe" -->
### Post-Processing

The post-processing is the same as in the IR bSSFP example. Only the reference values have been adjusted.
<!-- #endregion -->

```bash id="C6ce24MYpMR5"

# Post-Process Reconstruction

DIM=30
T1=0.8        # [s]
T2=1          # [s], initialization value! We constraint T2 to stay constant! -> pscale=1:1:0:1
FA_EFF=1.3333 # [s], Simulated FA (8 deg) / Reconstructed FA (6 deg)


# Create Reference Maps

bart phantom -c -x $DIM circ

bart scale -- $T1 circ t1ref
bart scale -- $T2 circ t2ref
bart scale -- $FA_EFF circ faref


# Resize output of Reconstruction
# -> compensate for 2-fold oversampling
bart resize -c 0 $DIM 1 $DIM reco reco_crop


# Convert and Mask Reconstructed Maps

bart slice 6 0 reco_crop r1map
bart spow -- -1 r1map _t1map
bart fmac _t1map circ t1map #mask
rm _t1map.cfl _t1map.hdr

bart slice 6 2 reco_crop r2map
bart spow -- -1 r2map _t2map
bart fmac _t2map circ t2map #mask
rm _t2map.cfl _t2map.hdr

bart slice 6 3 reco_crop _famap
bart fmac _famap circ famap #mask
rm _famap.cfl _famap.hdr
```

```python colab={"base_uri": "https://localhost:8080/", "height": 404} outputId="708fd603-f9f4-44b6-efc5-3011523dbdb7" id="tpeYIeIvpVpS"
diffplot('t1map', 't1ref', 2, 'viridis', 'T$_1$ / s')
diffplot('t2map', 't2ref', 0.2, 'copper', 'T$_2$ / s')
diffplot('famap', 'faref', 1.5, 'hot', 'FA/FA$_{nom}$')
```

<!-- #region id="rjiXOqt-sSBP" -->
The $T_1$ map only shows small errors to the reference.

The constrainting of the $T_2$ map by passing `pscale=1:1:0:1` to the `moba` call is visualized in the final $T_2$ map, because it has the same value it has been initialized with.

The reconstruction assumed a nominal flipangle of 6 degree, while the simulation was based on an 8 degree pulse. Thus, the reconstructed relative FA map is about 1.33 compensating the model difference.
<!-- #endregion -->

```bash id="bl5ItRth2N7Q"
rm *.{hdr,cfl}
```

```python id="45zeFlj52N7R"

```

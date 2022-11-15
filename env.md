# Environment
to run this project, I have created the `dis_ir` conda environment.
it is cloned from `rs_pttf` and then added the following:
```bash
pip install pytorch-lightning==1.7.7 
```

`rs_pttf` environment was created with the following script:
```
conda create -n rs_pttf python=3.9
conda activate rs_pttf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
echo $LD_LIBRARY_PATH
pip install --upgrade pip
pip install tensorflow
conda install -c anaconda jupyter
conda install -c conda-forge jupyter_contrib_nbextensions
conda install pandas scikit-learn
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
• Then we can install pytorch using the conda command :
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
• Installing other packages :
```
conda install -c conda-forge matplotlib
conda install tqdm
conda install ignite -c pytorch
pip install opencv-python seaborn
```
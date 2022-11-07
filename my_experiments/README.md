# My Experiments
this folder include various experiments with using the library (mostly with various datasets) 



# Installation
to install a conda env that can run this package do the following

```sh
conda create -n ptvae python=3.9
conda activate ptvae
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge cudnn=8.6.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
pip install pytorch-lightning==1.5.6
```

this should be enough to run the project



to build a docker image, go to the `sandbox` folder and:
```
docker build -f iDockerFile
```

to run the docker container, go to the parent folder of PyTorch-VAE and run:
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/PyTorch-VAE:/workspace/PyTorch-VAE -v </home/guy/sd1tb/datasets/CelebA>:/workspace/Data/CelebA <-v source_data_path:/workspace/Data/dest_data_folder> --rm -p 8888:8888 nvcr.io/nvidia/pytorch:22.07-py3_ptvae
```


from .CelebA import *
from .deep_fashion import *
from .fashion_dataset import *
from .oxford_pets import *
from .viton import *



datasets =   {'CelebA':MyCelebA,
              'DeepFashionICSR':DeepFashionISCR,
              'DeepFashionCATP':DeepFashionCATP,
              'OxfordPets':OxfordPets,
              'VITON':VITON,
              'FashionDataset':FashionDataset}

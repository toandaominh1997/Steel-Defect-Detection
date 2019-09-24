    .
    ├── configs
    │   └── train_config.yaml
    ├── data
    │   ├── test_images
    ├── datasets
    │   ├── __pycache__
    │   │   ├── augment.cpython-37.pyc
    │   │   ├── dataset.cpython-37.pyc
    │   │   └── __init__.cpython-37.pyc
    │   ├── dataset.py
    │   └── __init__.py
    ├── models
    │   ├── __pycache__
    │   │   ├── __init__.cpython-37.pyc
    │   │   ├── loss.cpython-37.pyc
    │   │   ├── losses.cpython-37.pyc
    │   │   ├── model_cls.cpython-37.pyc
    │   │   ├── model.cpython-37.pyc
    │   │   └── unet.cpython-37.pyc
    │   ├── __init__.py
    │   ├── losses.py
    │   ├── loss.py
    │   ├── model_cls.py
    │   ├── model.py
    │   ├── test.py
    │   └── unet.py
    ├── modules
    │   ├── lib
    │   │   ├── nn
    │   │   │   ├── modules
    │   │   │   │   ├── __pycache__
    │   │   │   │   │   ├── batchnorm.cpython-37.pyc
    │   │   │   │   │   ├── comm.cpython-37.pyc
    │   │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   │   └── replicate.cpython-37.pyc
    │   │   │   │   ├── tests
    │   │   │   │   │   ├── test_numeric_batchnorm.py
    │   │   │   │   │   └── test_sync_batchnorm.py
    │   │   │   │   ├── batchnorm.py
    │   │   │   │   ├── comm.py
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── replicate.py
    │   │   │   │   └── unittest.py
    │   │   │   ├── parallel
    │   │   │   │   ├── __pycache__
    │   │   │   │   │   ├── data_parallel.cpython-37.pyc
    │   │   │   │   │   └── __init__.cpython-37.pyc
    │   │   │   │   ├── data_parallel.py
    │   │   │   │   └── __init__.py
    │   │   │   ├── __pycache__
    │   │   │   │   └── __init__.cpython-37.pyc
    │   │   │   └── __init__.py
    │   │   └── utils
    │   │       ├── data
    │   │       │   ├── dataloader.py
    │   │       │   ├── dataset.py
    │   │       │   ├── distributed.py
    │   │       │   ├── __init__.py
    │   │       │   └── sampler.py
    │   │       ├── __init__.py
    │   │       └── th.py
    │   ├── __pycache__
    │   │   ├── hrnet.cpython-37.pyc
    │   │   ├── __init__.cpython-37.pyc
    │   │   ├── mobilenet.cpython-37.pyc
    │   │   ├── models.cpython-37.pyc
    │   │   ├── resnet.cpython-37.pyc
    │   │   ├── resnext.cpython-37.pyc
    │   │   ├── unet.cpython-37.pyc
    │   │   └── utils.cpython-37.pyc
    │   ├── segmentation_models
    │   │   ├── base
    │   │   │   ├── __pycache__
    │   │   │   │   ├── encoder_decoder.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   └── model.cpython-37.pyc
    │   │   │   ├── encoder_decoder.py
    │   │   │   ├── __init__.py
    │   │   │   └── model.py
    │   │   ├── common
    │   │   │   ├── __pycache__
    │   │   │   │   ├── blocks.cpython-37.pyc
    │   │   │   │   └── __init__.cpython-37.pyc
    │   │   │   ├── blocks.py
    │   │   │   └── __init__.py
    │   │   ├── encoders
    │   │   │   ├── __pycache__
    │   │   │   │   ├── densenet.cpython-37.pyc
    │   │   │   │   ├── dpn.cpython-37.pyc
    │   │   │   │   ├── inceptionresnetv2.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   ├── _preprocessing.cpython-37.pyc
    │   │   │   │   ├── resnet.cpython-37.pyc
    │   │   │   │   ├── senet.cpython-37.pyc
    │   │   │   │   └── vgg.cpython-37.pyc
    │   │   │   ├── densenet.py
    │   │   │   ├── dpn.py
    │   │   │   ├── inceptionresnetv2.py
    │   │   │   ├── __init__.py
    │   │   │   ├── _preprocessing.py
    │   │   │   ├── resnet.py
    │   │   │   ├── senet.py
    │   │   │   └── vgg.py
    │   │   ├── fpn
    │   │   │   ├── __pycache__
    │   │   │   │   ├── decoder.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   └── model.cpython-37.pyc
    │   │   │   ├── decoder.py
    │   │   │   ├── __init__.py
    │   │   │   └── model.py
    │   │   ├── linknet
    │   │   │   ├── __pycache__
    │   │   │   │   ├── decoder.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   └── model.cpython-37.pyc
    │   │   │   ├── decoder.py
    │   │   │   ├── __init__.py
    │   │   │   └── model.py
    │   │   ├── pspnet
    │   │   │   ├── __pycache__
    │   │   │   │   ├── decoder.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   └── model.cpython-37.pyc
    │   │   │   ├── decoder.py
    │   │   │   ├── __init__.py
    │   │   │   └── model.py
    │   │   ├── __pycache__
    │   │   │   └── __init__.cpython-37.pyc
    │   │   ├── unet
    │   │   │   ├── __pycache__
    │   │   │   │   ├── decoder.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   └── model.cpython-37.pyc
    │   │   │   ├── decoder.py
    │   │   │   ├── __init__.py
    │   │   │   └── model.py
    │   │   ├── utils
    │   │   │   ├── __pycache__
    │   │   │   │   ├── functions.cpython-37.pyc
    │   │   │   │   ├── __init__.cpython-37.pyc
    │   │   │   │   ├── losses.cpython-37.pyc
    │   │   │   │   ├── metrics.cpython-37.pyc
    │   │   │   │   └── train.cpython-37.pyc
    │   │   │   ├── functions.py
    │   │   │   ├── __init__.py
    │   │   │   ├── losses.py
    │   │   │   ├── metrics.py
    │   │   │   └── train.py
    │   │   ├── __init__.py
    │   │   └── __version__.py
    │   ├── hrnet.py
    │   ├── __init__.py
    │   ├── mobilenet.py
    │   ├── models.py
    │   ├── resnet.py
    │   ├── resnext.py
    │   ├── test_modules.py
    │   └── utils.py
    ├── optimizers
    │   ├── __pycache__
    │   │   ├── __init__.cpython-37.pyc
    │   │   └── optimizer.cpython-37.pyc
    │   ├── __init__.py
    │   └── optimizer.py
    ├── __pycache__
    │   ├── learning.cpython-37.pyc
    │   ├── parse_config.cpython-37.pyc
    │   ├── pytorch_modelsize.cpython-37.pyc
    │   ├── segmentation.cpython-37.pyc
    │   ├── test.cpython-37.pyc
    │   └── utils.cpython-37.pyc
    ├── saved
    ├── utils
    │   ├── __pycache__
    │   │   ├── helper.cpython-37.pyc
    │   │   ├── __init__.cpython-37.pyc
    │   │   ├── meter.cpython-37.pyc
    │   │   ├── metric.cpython-37.pyc
    │   │   └── util.cpython-37.pyc
    │   ├── helper.py
    │   ├── __init__.py
    │   └── meter.py
    ├── learning.py
    ├── main.json
    ├── requirements.txt
    ├── sm.py
    └── train.py

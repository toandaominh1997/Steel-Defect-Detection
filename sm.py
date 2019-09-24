from sagemaker.pytorch import PyTorch
import random
instance_type = 'ml.p3.2xlarge'
# train_data_path = 's3://rnd-ocr/quan/data.zip'
train_data_path = 's3://rnd-ocr/kan/quan.txt'
output_path = 's3://rnd-ocr/hades/output'
code_location = 's3://rnd-ocr/hades/demo'
role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
source_dir = "."
name_list = ["nick","hades","kizd","vanle","nick","myl","kyocera","erikson","ken"]
# name_list = ["vanle-segmentation"]
random.shuffle(name_list)
print(">>> start <<<")


pytorch_estimator = PyTorch(entry_point='train.py',
                           source_dir=source_dir,
                           code_location=code_location,
                           output_path=output_path,
                           role=role,
                           train_use_spot_instances=True,
                           train_max_wait=10*86400+20,
                           train_instance_type=instance_type,
                           train_instance_count=1,
                           train_volume_size=300,
                           base_job_name= name_list[0]+"-test",
                           train_max_run=10*86400,
                           framework_version='1.1.0',
                           py_version="py3")
# pytorch_estimator.
pytorch_estimator.fit() 

from sagemaker.pytorch import PyTorch

# ref: https://aws.amazon.com/sagemaker/pricing/instance-types/
instance_type = 'ml.p3.2xlarge'
# DATASET/ForFFG/HW_Printed_fixform/synthesis04
train_data_path = 's3://rnd-ocr/DATASET/ForInvoice/'
output_path = 's3://rnd-ocr/anh/demo/output/'
code_location = 's3://rnd-ocr/anh/demo/sourcecode'
role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
source_dir = "."
pytorch_estimator = PyTorch(entry_point='train.py',
                            source_dir=source_dir,
                            code_location=code_location,
                            output_path=output_path,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            train_volume_size=200,
                            base_job_name= 'toyota4-u',
                            train_max_run=5*86400,  # 86400s ~ 1day
                            framework_version='1.1.0',
                            py_version="py3")

pytorch_estimator.fit() 

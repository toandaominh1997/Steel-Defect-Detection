from sagemaker.pytorch import PyTorch
import argparse 

parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--encoder', default="resnet50", type=str)
parser.add_argument('--decoder', default="Unet", type=str)  
parser.add_argument('--encoder_weights', default="imagenet", type=str) 
args = parser.parse_args()

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
                            base_job_name= 'toyota4',
                            train_max_run=5*86400,  # 86400s ~ 1day
                            framework_version='1.1.0',
                            py_version="py3",
                            hyperparameters = {'encoder': "{}".format(args.encoder), 'decoder': "{}".format(args.decoder), 'batch_size': args.batch_size, "encoder_weights": "{}".format(args.encoder_weights)}
                            )

pytorch_estimator.fit() 

import os
import boto3
import requests

# Environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET')
AWS_REGION = os.getenv('AWS_REGION')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')
API_URL = os.getenv('API_URL')
HF_API_KEY = os.getenv('HF_API_KEY')

# Set up S3 client
s3 = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# API endpoint
headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def check_s3(download_image=False, download_image_name=None):
    # List all objects in S3 bucket
    response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET)

    for obj in response.get('Contents', []):
        print(obj['Key'])

    if not download_image:
        return

    # download image from S3    
    # Get file directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    temp_dir = os.path.join(current_dir, 'tmp')
    os.makedirs(os.path.dirname(temp_dir), exist_ok=True)

    print(temp_dir)

    s3.download_file(AWS_S3_BUCKET, f'{download_image_name}.jpg', f'{temp_dir}/{download_image_name}.jpg')
    s3.download_file(AWS_S3_BUCKET, f'{download_image_name}_heatmap.jpg', f'{temp_dir}/{download_image_name}_heatmap.jpg')
    
def upload_image(image_name):
    # Get file directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    temp_dir = os.path.join(current_dir, 'tmp')
    os.makedirs(os.path.dirname(temp_dir), exist_ok=True)

    # Upload image to S3
    s3.upload_file(f'{temp_dir}/{image_name}.jpg', AWS_S3_BUCKET, f'{image_name}.jpg')
    
def test_endpoint(image_name):
    def query(payload):
        print("Local payload:", payload)
        resp = requests.post(API_URL, headers=headers, json=payload)
        print("Sent JSON:", resp.request.body)
        return resp.json()

    output = query({
        "inputs": {
            "image_name": f"{image_name}.jpg"
        }
    })

    print("Output:", output)

test_endpoint('AI_Magazine')
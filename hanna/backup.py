import boto3
from botocore.exceptions import ClientError
import json
import os
from django.conf import settings


class AWSBackup:
    def __init__(self, bucket_name: str):
        self.aws_client = boto3.client('s3',
                                        aws_access_key_id=settings.AWS_ACCESS_KEY,
                                        aws_secret_access_key=settings.AWS_SECRET_KEY,
                                        region_name='us-east-2')

        self.bucket_name = bucket_name

    def verify_file_ext(self, name: str) -> bool:
        allowed = ["docx", "txt", "pdf", "pptx", "xlsx", "csv"]
        file = name.split('.')

        if file[-1] in allowed:
            return True

        return False

    def download_object_file(self, name: str) -> bool:
        try:
            self.aws_client.download_file(Filename="./_tmp/" + name, Bucket=self.bucket_name, Key=name)
            return True
        except Exception as e:
            print("BACKUP DOWNLOAD FILE: ")
            print(e)
            return False

    def get_bucket_list(self) -> str:
        try:
            response = self.aws_client.list_buckets()
            print(response)
            return response
        except Exception as e:
            print("BACKUP GET BUCKET LIST: ")
            print(e)
            return ""

    def get_object_list(self):
        response = self.aws_client.list_objects_v2(Bucket=settings.BUCKET_NAME)
        obs = []

        # Print the list of object keys (names)
        if 'Contents' in response:
            for obj in response['Contents']:
                obs.append(obj['Key'])
            return obs
        else:
            return None

    def create_bucket(self, bucket_name: str, region: str = 'us-east-2') -> bool:
        try:
            location = {'LocationConstraint': region}
            self.aws_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
            print(f"Bucket {bucket_name} created successfully.")
        except ClientError as e:
            print(f"BACKUP CREATE BUCKET:")
            print(e)
            return False
        return True

    def read_object(self, object_name: str) -> str or None:
        try:
            # Fetch the object from S3
            response = self.aws_client.get_object(Bucket=self.bucket_name, Key=object_name)

            # Read the content of the object
            content = json.loads(response['Body'].read().decode('utf-8'))

            return content
        except Exception as e:
            print(f"BACKUP READ OBJECT:")
            print(e)
            return None

    def upload_file(self, file_name: str, bucket: str, object_name: str = None) -> None or str:
        if object_name is None:
            object_name = file_name

        try:
            self.aws_client.upload_file(file_name, bucket, object_name)
            print(f"File {file_name} uploaded successfully to {bucket}/{object_name}")
        except ClientError as e:
            print(f"Error: {e}")
            return False
        return True

    def delete_object(self, bucket_name: str, object_name: str):
        try:
            self.aws_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"Object '{object_name}' deleted successfully from bucket '{bucket_name}'.")
        except ClientError as e:
            print(f"Error: {e}")

    def upload_json(self, object_name: str, data: dict) -> bool:
        json_data = json.dumps(data)

        try:
            # Upload the JSON string to S3
            self.aws_client.put_object(Bucket=self.bucket_name, Key=object_name, Body=json_data, ContentType='application/json')
            print(f"JSON data uploaded successfully to {self.bucket_name}/{object_name}")
            return True
        except Exception as e:
            print(f"BACKUP UPLOAD JSON:")
            print(e)
            return False

    def delete_bucket(self, bucket_name: str) -> bool:
        try:
            self.aws_client.delete_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' deleted successfully.")
            return True
        except ClientError as e:
            print(f"BUCKET DELETE BUCKET:")
            print(e)
            return False

import json
import boto3
import botocore
from datetime import datetime
import base64

def handle_text_to_image(event, context):

    # Extracting input from the API gateway
    input_data = json.loads(event['body'])
    
    # The input text to generate the image from
    input_text = input_data['inputText']

    image_generator = boto3.client("runtime.sagemaker",region_name="us-east-1",config=botocore.config.Config(read_timeout=310, retries={'max_attempts': 2}))
    
    # S3 client for image storage
    storage_client = boto3.client('s3')

    generation_params = {
        "prompt_details": [{"text": input_text}],
        # Influence of the prompt on the generated image
        "generation_impact": 12,
        "random_seed": 42, # To allow reproducibility
        "generation_steps": 120
    }

    model_response = image_generator.invoke_endpoint(EndpointName='text-image-model', Body=json.dumps(generation_params),ContentType="application/json",Accept="application/json")
    
    # Processing the model response to extract and decode the image
    model_output = json.loads(model_response['Body'].read())
    encoded_image_str = model_output["generated_images"][0]["image_base64"]
    decoded_image = base64.b64decode(encoded_image_str)

    # S3 bucket and key for image upload
    target_bucket = 'generated-images-bucket'
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    image_file_name = f"generated_images/{current_time}.png"

    storage_client.put_object(Bucket=target_bucket, Key=image_file_name, Body=decoded_image, ContentType='image/png')

    return {
        'statusCode': 200,
        'body': json.dumps('Image successfully saved to S3')
    }

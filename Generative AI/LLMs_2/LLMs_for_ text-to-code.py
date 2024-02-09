import boto3
import botocore.config
import json
from datetime import datetime

def generate_code_from_bedrock(message: str, language: str) -> str:
    """
    Generates code based on the provided message and language using the Bedrock AI model.
    
    Parameters:
    message (str): Instructions for code generation.
    language (str): Programming language for the generated code.
    
    Returns:
    str: The generated code snippet.
    """
    input_text = f"""
Human: Write {language} code for the following instructions: {message}. Just write the code, don't use examples.
Assistant:
    """
    request_body = {
        "prompt": input_text,
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_k": 250,
        "top_p": 0.2,
        "stop_sequences": ["\n\nReady for New Prompt:"]
    }

    try:
        # Bedrock client configuration
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3})
        )
        response = bedrock_client.invoke_model(body=json.dumps(request_body), modelId="ai21.labs.jurassic-2")
        response_content = response.get('body').read().decode('utf-8')
        response_data = json.loads(response_content)
        generated_code = response_data["completion"].strip()
        return generated_code

    except Exception as ex:
        print(f"An error occurred during code generation: {ex}")
        return ""

def save_code_to_s3(code: str, bucket_name: str, object_key: str):
    """
    Saves the generated code to an S3 bucket.
    
    Parameters:
    code (str): The generated code to save.
    bucket_name (str): Name of the S3 bucket.
    object_key (str): Object key in the S3 bucket.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=code)
        print("Successfully saved generated code to S3.")
    except Exception as ex:
        print(f"An error occurred when saving the code to S3: {ex}")

def lambda_handler(event, context):
    """
    The AWS Lambda function handler.
    
    Parameters:
    event: The event dict containing the details for code generation.
    context: The context in which the Lambda function is called.
    
    Returns:
    dict: A response object with status code and body.
    """
    event_body = json.loads(event['body'])
    message = event_body['message']
    language = event_body['key']
    print(f"Message: {message}, Language: {language}")

    generated_code = generate_code_from_bedrock(message, language)

    if generated_code:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        object_key = f'output/{timestamp}.py'
        bucket_name = 'my-bucket-for-code-generation'
        save_code_to_s3(generated_code, bucket_name, object_key)
    else:
        print("No code was generated.")

    return {
        'statusCode': 200,
        'body': json.dumps("Code generation complete.")
    }

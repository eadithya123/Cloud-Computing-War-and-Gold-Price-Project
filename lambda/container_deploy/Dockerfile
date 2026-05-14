FROM public.ecr.aws/lambda/python:3.11

RUN pip install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2 pandas==2.1.4 joblib textblob feedparser yfinance boto3

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]

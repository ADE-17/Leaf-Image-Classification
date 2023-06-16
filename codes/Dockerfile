FROM pytorch/pytorch:latest

COPY inference.py /codes/inference.py
COPY model_epoch_20.pth Data/model_epoch_20.pth

COPY requirements.txt /codes/requirements.txt

RUN pip install -r /codes/requirements.txt

EXPOSE $PORT

RUN gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
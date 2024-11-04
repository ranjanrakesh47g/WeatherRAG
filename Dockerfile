FROM python:3.12
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python3", "src/service.py"]
EXPOSE 8000

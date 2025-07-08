FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt wordnet
COPY . .
EXPOSE 5000

CMD ["flask", "--app", "api", "run", "--host=0.0.0.0"]
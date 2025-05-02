# 1. Base Python Image
FROM python:3.10

# 2. Set working directory
WORKDIR /app

# 3. Copy only requirements.txt first
COPY requirements.txt .

# 4. Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# 5. Install dependencies from requirements.txt
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu torchvision==0.15.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt


# 6. Clean pip cache manually (to keep final image small)
RUN rm -rf ~/.cache/pip

# 7. Copy the rest of your app code
COPY . .

# 8. Command to run your app
CMD ["python", "app.py"]

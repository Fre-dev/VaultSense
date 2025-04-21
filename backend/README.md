# VaultSense OCR API

A FastAPI backend for OCR processing using Mistral AI API.

## Features

- Extract structured information from images using Mistral AI's OCR capabilities
- Process various image formats (JPEG, PNG, TIFF, BMP, GIF)
- Return structured JSON data extracted from documents

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip3 install -r requirements.txt
   ```
4. Create a `.env` file in the `backend` directory using `.env.example` as a template:
   ```
   cp .env.example .env
   ```
5. Edit the `.env` file and add your Mistral AI API key

## Running the Application

```
python server.py
```

The API will be available at `http://localhost:8001`.

## API Endpoints

### Health Check

```
GET /health
```

Returns the status of the API.

### Process Image

```
POST /api/v1/ocr/process-image
```

Upload an image to extract structured information.

**Request:**
- Form data with a file parameter named `file`

**Response:**
- JSON object with structured data extracted from the image

## API Documentation

Once the server is running, you can access the Swagger UI documentation at:

```
http://localhost:8001/docs
```

Or the ReDoc documentation at:

```
http://localhost:8001/redoc
``` 
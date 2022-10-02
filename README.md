# Backend

The backend is built using Flask and various machine learning libraries (eg. PyTorch and Tensorflow). The server will be running on `http://localhost:8080` by default.

## Running the Server (with conda)

To run the server with conda, install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).


```bash
conda env create -f environment.yaml
conda activate DLW
python3 main.py
```
## Running the server (with pip)
```bash
pip install -r requirements.txt
```
## Project Directories
- `./data`: Data Sources required for our ML models
- `./models`: ML Models and files (ANN, Google DeepLab v3+, MinMaxScaler)
- `./scripts`: Miscellaneous scripts to run on our server
- `constants.py`: Constants used within our server
- `main.py`: Entry Point to our flask application
- `processing.py`: Miscellaneous helper functions for processing purposes

## Endpoints

#### `GET /`

`Description`: Health check to ensure that the server is up and running.

Request Body:

```
-
```
Response:
```
Health Check
```

#### `GET /upload`
`Description`: Ensure that the user POSTS to the endpoint.

Request body:
```
-
```
Response:
```
Request Method not supported
```

#### `POST /upload`
`Description`: Retrieves bikescore and walkscore from the deployed ML model.

Request body:
```JSON
{
  "file": FILE,
  "coordinates": "1.2836322,103.8461028"
}
```
Response:
```JSON
{
  "bikescore": 90.102301,
  "image":"2fanjksdnasjkdnsaj...",
  "walkscore": 82.213212
}
```

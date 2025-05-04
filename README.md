# Bank Marketing Prediction API

A FastAPI-based REST API for predicting customer subscription to a bank financial product using a CatBoost machine learning model.

## Overview

This application serves a machine learning model that predicts the likelihood of a customer subscribing to a bank financial product. The prediction is based on various customer attributes and economic indicators.

The API is containerized using Docker and can be easily deployed using Docker Compose.

## Model Information

The prediction service uses a CatBoost classifier, an efficient implementation of gradient boosting on decision trees. 

Download **catboost_model.cbm** from [Google Drive](https://drive.google.com/file/d/142trrBb--OJh-N11n_CrtzGQbJtWML-T/view?usp=sharing) and place it in the *app* directory.

## Features

- High-performance predictions using CatBoost
- Comprehensive input validation
- Feature engineering pipeline integrated into the API
- Health check endpoint for monitoring
- Model information endpoint for transparency
- Both single and batch prediction endpoints
- Containerized for easy deployment
- Resource limiting for predictable performance

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /model-info` - Information about the loaded model
- `POST /predict` - Make predictions for one or more customers
- `POST /batch-predict` - Alias for predict, optimized for batch processing

## Input Data Schema

The API accepts customer data with the following attributes:

- `age`: Customer age (18-100)
- `occupation`: Type of job/occupation
- `marital_status`: Marital status
- `education`: Education level
- `housing_loan`: Has housing loan (yes/no)
- `personal_loan`: Has personal loan (yes/no) 
- `contact_mode`: Contact communication mode
- `month`: Last contact month (3-letter code)
- `week_day`: Last contact day of the week (3-letter code)
- `last_contact_duration`: Last contact duration in seconds
- `contacts_per_campaign`: Number of contacts during this campaign
- `previous_outcome`: Outcome of previous campaign
- `emp_var_rate`: Employment variation rate
- `cons_price_index`: Consumer price index
- `euri_3_month`: Euribor 3 month rate

## Setup and Deployment

### Prerequisites

- Docker
- The CatBoost model file (`catboost_model.cbm`)

### Environment Variables

Create a `.env` file with the following variables:

```
MODEL_PATH=catboost_model.cbm
PREDICTION_THRESHOLD=0.487
LOG_LEVEL=INFO
```

### Running with Docker Compose

1. Clone this repository
2. Download the model file and place it in the `app` directory
3. Run `docker-compose up -d`
4. The API will be available at http://localhost:8000

## Example Usage

### Request

```json
{
  "customers": [
    {
      "age": 41,
      "occupation": "technician",
      "marital_status": "married",
      "education": "university.degree",
      "housing_loan": "yes",
      "personal_loan": "no",
      "contact_mode": "cellular",
      "month": "may",
      "week_day": "mon",
      "last_contact_duration": 180,
      "contacts_per_campaign": 2,
      "previous_outcome": "success",
      "emp_var_rate": -0.1,
      "cons_price_index": 93.2,
      "euri_3_month": 0.6
    }
  ]
}
```

### Response

```json
{
  "predictions": [
    {
      "customer_index": 0,
      "probability": 0.72,
      "prediction": 1,
      "prediction_label": "yes"
    }
  ],
  "model_version": "catboost_model.cbm",
  "threshold_used": 0.487,
  "processing_time_ms": 15.3
}
```

## Development

For development, you can mount the app directory as a volume to enable hot reloading by modifying the volumes section in docker-compose.yml to use the development volume mapping.

1. Clone the repository:
   ```bash
   git clone git@github.com:timofeytkachenko/2021.ai_test_task_REST_API.git
   ```
2. Navigate to the project directory:
   ```bash
   cd 2021.ai_test_task_REST_API
   ```
3. Run Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Performance Considerations

- The API is configured to use 1 CPU core and 1GB of memory by default
- For higher throughput, adjust the resource limits in docker-compose.yml
- The number of workers can be modified in the Dockerfile CMD

## Contributing

Author: [Timofey Tkachenko](https://linktr.ee/timofey_tkachenko)
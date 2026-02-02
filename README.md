# Telco Churn – End-to-End ML Project
  -----------------------------------

  **Live Demo:** http://54.211.9.165:8080/ui/


  ## Goal

Designed and deployed an end-to-end machine learning system to predict telecom customer churn, covering data preparation, model training, and production deployment with an API and web interface on AWS.

## Problem solved & benefits

* **Faster decision-making:** Identifies customers at risk of churn early, enabling proactive retention actions.

* **Production-ready ML:** The model is exposed through a REST API and a lightweight UI, allowing easy testing without notebooks.

* **Reliable deployments:** Containerization and CI/CD ensure consistent builds, testing, and redeployments for every change.

* **Experiment traceability:** MLflow logs experiments, metrics, and artifacts to support reproducibility and auditability.

## Core Build & Features

* **Data & Modeling:** Performed feature engineering and trained an XGBoost classifier, with experiments logged in MLflow.
* **Model Tracking:** Captured runs, metrics, and serialized models under a dedicated MLflow experiment.
* **Inference Service:** Built a FastAPI service exposing a /predict POST endpoint and a root / health check.
* **Web Interface:** Added a Gradio-based UI at /ui for quick, shareable manual testing.
* **Containerization:** Packaged the application into a Docker image with a Uvicorn entrypoint (src.app.main:app) running on port 8080.
* **CI/CD:** Implemented GitHub Actions to build and push images to Docker Hub, with optional ECS service redeployments.
* **Orchestration:** Deployed the container on AWS ECS using Fargate for serverless execution.
* **Networking:** Configured an Application Load Balancer (HTTP:80) to route traffic to an IP-based target group on port 8080.


## Deployment flow (high-level)

* A push to the main branch triggers GitHub Actions to build the Docker image and push it to Docker Hub.
* The ECS service is then updated—either manually or through the workflow—to initiate a new deployment.
* The Application Load Balancer performs health checks on the / endpoint over port 8080 and routes traffic once the service is healthy.
* End users can send requests to the /predict endpoint or access the Gradio UI at /ui via the ALB DNS.

## Problems Faced and How They Were Resolved


### Unhealthy Targets Behind Application Load Balancer
**Problem:**  
The ECS tasks were repeatedly marked as unhealthy by the Application Load Balancer, preventing traffic from being routed to the service.

**Root Causes:**  
- Missing health check endpoint in the application  
- Port mismatch between the ALB target group and the container  

**Solution:**  
- Added a root `GET /` health-check endpoint in the FastAPI application  
- Verified that the ALB listener on port `80` correctly forwards traffic to the target group on port `8080`  
- Updated the target group health check path to `/`  

---

### Module Import Error Inside Docker Container
**Problem:**  
The application failed to start inside the container with a `ModuleNotFoundError`.

**Root Cause:**  
- Python could not resolve the `src/` directory during runtime inside the Docker container  

**Solution:**  
- Set `PYTHONPATH=/app/src` in the Dockerfile  
- Corrected the Uvicorn entrypoint to `src.app.main:app`  

---

### ECS Service Not Picking Up the Latest Image
**Problem:**  
After pushing a new Docker image, the ECS service continued running the old version.

**Root Cause:**  
- ECS service was still using the previous task definition revision  

**Solution:**  
- Forced a new ECS deployment after pushing the updated image to the registry  
- Ensured the task definition referenced the latest image tag  

---

### Local vs Production Path Inconsistencies
**Problem:**  
The application worked locally but failed when deployed to ECS due to path resolution issues.

**Root Cause:**  
- MLflow artifact URIs and file paths differed between local and containerized environments  

**Solution:**  
- Used direct local artifact paths during development  
- Ensured the container loads the packaged model path at runtime in production  


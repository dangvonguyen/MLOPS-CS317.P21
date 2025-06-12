import json
import logging
import os
import subprocess
import time

from fastapi import BackgroundTasks, FastAPI, Request

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("retraining-service")

app = FastAPI(title="Model Retraining Service")

MODEL_DIR = "model"
TRAINING_SCRIPT = "src/train.py"
MAX_BACKUPS = 5


def run_retraining():
    """Background task to retrain the model"""
    logger.info("Starting model retraining...")
    try:
        # Timestamp for backup
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Backup current model
        if os.path.exists(f"{MODEL_DIR}/sentiment_model.joblib"):
            backup_path = f"{MODEL_DIR}/sentiment_model_{timestamp}.joblib"
            os.rename(f"{MODEL_DIR}/sentiment_model.joblib", backup_path)
            logger.info(f"Backed up current model to {backup_path}")

        # Run training script
        try:
            result = subprocess.run(
                ["python", TRAINING_SCRIPT],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info("Retraining completed successfully")
            logger.debug(f"Training output: {result.stdout}")

            # Clean up old backups
            cleanup_old_backups()

        except subprocess.CalledProcessError as e:
            logger.error(f"Retraining failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            if e.stdout:
                logger.error(f"Standard output: {e.stdout}")

            # Restore backup if retraining failed
            restore_backup()
            raise

    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}")
        restore_backup()


def cleanup_old_backups():
    """Clean up old model backups, keeping only the most recent ones"""
    try:
        backup_files = [
            f
            for f in os.listdir(MODEL_DIR)
            if f.startswith("sentiment_model_") and f.endswith(".joblib")
        ]
        if len(backup_files) > MAX_BACKUPS:
            backup_files.sort(reverse=True)
            for old_backup in backup_files[MAX_BACKUPS:]:
                os.remove(os.path.join(MODEL_DIR, old_backup))
                logger.info(f"Removed old backup: {old_backup}")
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}")


def restore_backup():
    """Restore the model from the most recent backup"""
    try:
        backup_files = [
            f
            for f in os.listdir(MODEL_DIR)
            if f.startswith("sentiment_model_") and f.endswith(".joblib")
        ]
        if backup_files:
            backup_files.sort(reverse=True)
            latest_backup = backup_files[0]
            os.rename(
                os.path.join(MODEL_DIR, latest_backup),
                os.path.join(MODEL_DIR, "sentiment_model.joblib"),
            )
            logger.info(f"Restored model from backup: {latest_backup}")
    except Exception as e:
        logger.error(f"Error restoring backup: {str(e)}")


@app.get("/manual-retrain")
async def manual_retrain(background_tasks: BackgroundTasks):
    """
    Endpoint to manually trigger model retraining
    """
    logger.info("Manual retraining triggered via GET endpoint")
    background_tasks.add_task(run_retraining)
    return {"status": "retraining_scheduled", "triggered_by": "manual request"}


@app.post("/trigger-retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, request: Request):
    """
    Endpoint that receives alerts from Alertmanager and triggers model retraining
    """
    try:
        alert_data = await request.json()
        logger.info("Received retraining trigger from Alertmanager")
        logger.debug(f"Alert data: {json.dumps(alert_data, indent=2)}")

        # Extract alert information
        alerts = []
        if "alerts" in alert_data:
            for alert in alert_data["alerts"]:
                alert_name = alert.get("labels", {}).get("alertname", "Unknown")
                severity = alert.get("labels", {}).get("severity", "Unknown")
                alerts.append(f"{alert_name} (severity: {severity})")

        alert_summary = ", ".join(alerts) if alerts else "No specific alerts"
        logger.info(f"Triggering retraining due to alerts: {alert_summary}")

        # Add retraining task to background tasks
        background_tasks.add_task(run_retraining)

        return {"status": "retraining_scheduled", "triggered_by": alert_summary}

    except Exception as e:
        logger.error(f"Error processing alert: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("retraining_service:app", host="0.0.0.0", port=8080)

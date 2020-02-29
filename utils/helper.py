import os


def get_temp_data_bucket():
    """Check TEMP_DATA_BUCKET ends with /."""
    TEMP_DATA_BUCKET=os.getenv("TEMP_DATA_BUCKET")

    if not TEMP_DATA_BUCKET.endswith("/"):
        return TEMP_DATA_BUCKET + "/"
    else: 
        return TEMP_DATA_BUCKET

import os

import requests


def download_artefact(pipeline_run_id: str):
    print(f"Downloading artefact for pipeline run: {pipeline_run_id}")
    filename = f"/tmp/{pipeline_run_id}-artefact.zip"
    with requests.get(
        f"https://api.amoy.ai/v1/artefact/{pipeline_run_id}/internal",
        headers={"X-Bedrock-Api-Token": os.environ["BEDROCK_API_TOKEN"]},
        stream=True,
    ) as response:
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    return filename


def download_artefact_from_latest_run(pipeline_public_id: str):
    response = requests.get(
        f"https://api.amoy.ai/v1/training_pipeline/{pipeline_public_id}/run/",
        headers={"X-Bedrock-Access-Token": os.environ["BEDROCK_ACCESS_TOKEN"]},
        timeout=30,
    )
    response.raise_for_status()
    runs = response.json()
    if not runs:
        print(f"No runs to download: {pipeline_public_id}")
        return
    last_run = max(runs, key=lambda run: run["created_at"])
    filename = download_artefact(last_run["entity_id"])
    print(f"Downloaded: {filename}")

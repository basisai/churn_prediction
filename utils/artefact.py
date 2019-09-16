import glob
import os
import zipfile
from typing import Optional

import requests


def download_artefact_by_run_id(
    pipeline_run_id: str, output_filepath: Optional[str] = None
) -> str:
    """
    Downloads the model artefact produced by a specific pipeline run to a local directory,
    defaults to `/tmp/{pipeline_run_id}-artefact.zip`.

    :param pipeline_run_id: The public id of the pipeline run to download from
    :type pipeline_run_id: str
    :return: The file path that the artefact is saved to
    :rtype: str
    """

    print(f"Downloading artefact for pipeline run: {pipeline_run_id}")
    # Call Bedrock API to get the download url of a model artefact by its pipeline run id
    response = requests.get(
        # BEDROCK_API_DOMAIN is automatically injected into the workload environment by Bedrock
        url=f"{os.environ['BEDROCK_API_DOMAIN']}/v1/artefact/{pipeline_run_id}",
        # The access token is a long lived token generated from "API tokens" page on Bedrock UI.
        # BEDROCK_ACCESS_TOKEN must be declared as a pipeline secret in bedrock.hcl and saved as
        # a default value in pipeline settings page on Bedrock UI.
        headers={"X-Bedrock-Access-Token": os.environ["BEDROCK_ACCESS_TOKEN"]},
        # Times out the request if no reply is received within 30 seconds
        timeout=30,
    )
    # Verify that the API call is successful
    response.raise_for_status()
    artefact = response.json()

    # Save the downloaded file in chunks to reduce memory usage when downloading large artefacts
    downloaded = requests.get(url=artefact["download_url"], stream=True)
    filename = output_filepath or f"/tmp/{pipeline_run_id}-artefact.zip"
    with open(filename, "wb") as output:
        # Choose a chunk size in multiples of page size (4KB)
        for chunk in downloaded.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                output.write(chunk)

    return filename


def download_artefact_from_latest_run(
    pipeline_public_id: str, output_filepath: Optional[str] = None
) -> Optional[str]:
    """
    Downloads the model artefact produced by the latest successful run of a given pipeline. The output
    filepath defaults to `/tmp/{pipeline_run_id}-artefact.zip`.

    :param pipeline_public_id: The public id of the pipeline to download from
    :type pipeline_public_id: str
    :return: The file path that the artefact is saved to
    :rtype: str
    """

    # Call Bedrock API to get all runs of the training pipeline
    response = requests.get(
        url=f"{os.environ['BEDROCK_API_DOMAIN']}/v1/training_pipeline/{pipeline_public_id}/run/",
        headers={"X-Bedrock-Access-Token": os.environ["BEDROCK_ACCESS_TOKEN"]},
        timeout=30,
    )
    # Verify that the API call successfully returns a json array
    response.raise_for_status()
    runs = response.json()

    # Filter by creation time for the last successful run
    successful_runs = filter(lambda run: run["status"] == "Succeeded", runs)
    try:
        last_run = max(successful_runs, key=lambda run: run["created_at"])
    except ValueError as exc:
        raise Exception(
            f"No successful runs found for pipieline: {pipeline_public_id}"
        ) from exc

    # Calls Bedrock API again to download artefact from the latest run
    filename = download_artefact_by_run_id(
        pipeline_run_id=last_run["entity_id"], output_filepath=output_filepath
    )
    print(f"Downloaded artefact: {filename}")
    return filename


def download_and_unzip_artefact(output_directory: Optional[str] = None) -> bool:
    """
    Downloads and unzips the model artefact of the last successful run of the current pipeline.
    The default unzip path is `/tmp/{pipeline_id}`.

    If this is the first run, there will be no past runs to download artefacts from. This function
    will return False and users should handle this case by training from scratch. Once Bedrock
    supports manual uploading of model artefacts, we will be able to better handle this use case
    by downloading directly from the model store.

    :param output_directory: Path that downloaded artefacts will be unzipped to
    :type output_directory: Optional[str]
    :return: Whether artefacts were successfully downloaded and unzipped
    :rtype: bool
    """

    # Currently users have to declare the public id of this pipeline in bedrock.hcl as one of the
    # training parameters. In the future, we may look into better supporting this use case by
    # injecting the public id automatically into the workload environment.
    pipeline_id = os.getenv("PIPELINE_PUBLIC_ID")
    if not pipeline_id:
        # This environment variable should be left as an empty string for the initial run so that
        # training begins without downloading any previous artefacts. Subsequently, users may reset
        # this parameter to an empty string to retrain from scratch.
        return False

    # Download artefact from the last successful pipeline run
    filename = download_artefact_from_latest_run(pipeline_public_id=pipeline_id)

    # Extract the downloaded artefacts
    output_directory = output_directory or f"/tmp/{pipeline_id}"
    with zipfile.ZipFile(filename, "r") as f:
        f.extractall(path=output_directory)

    # List the contents for manual verification
    print(glob.glob(f"{output_directory}/*"))
    return True

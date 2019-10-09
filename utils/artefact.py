import glob
import os

from typing import Optional
from bdrk.v1_util import download_and_unzip_artefact, get_artefact_stream
from bdrk.v1 import ApiClient, Configuration, ModelApi, PipelineApi, ServeApi

configuration = Configuration()

# This is your personal access token https://docs.basis-ai.com/getting-started/rest-apis/personal-api-token
configuration.api_key["X-Bedrock-Access-Token"] = os.environ["BEDROCK_ACCESS_TOKEN"]
configuration.host = os.environ.get("BEDROCK_API_DOMAIN", "https://api.bdrk.ai")

api_client = ApiClient(configuration)
pipeline_api = PipelineApi(api_client)


def download_artefact_by_run_id(
    pipeline_id: str,
    pipeline_run_id: str,
    output_filepath: Optional[str] = None
) -> str:
    """
    Downloads the model artefact produced by a specific pipeline run to a local directory,
    defaults to `/tmp/{pipeline_run_id}-artefact.zip`.

    :param pipeline_id: The public id of the pipeline to download from
    :type pipeline_id: str
    :param pipeline_run_id: The public id of the pipeline run to download from
    :type pipeline_run_id: str
    :param output_filepath: Output file name with path
    :type output_filepath: str
    :return: The file path that the artefact is saved to
    :rtype: str
    """

    print(f"Downloading artefact for pipeline {pipeline_id} and run: {pipeline_run_id}")
    pipeline = pipeline_api.get_training_pipeline_by_id(pipeline_id=pipeline_id)
    run = pipeline_api.get_training_pipeline_run(pipeline_id=pipeline_id, run_id=pipeline_run_id)

    stream = get_artefact_stream(
        api_client=api_client,
        model_id=pipeline.model_id,
        model_artefact_id=run.artefact_id
    )
    filename = output_filepath or f"/tmp/{pipeline_run_id}-artefact.zip"
    with open(filename, "wb") as output:
        output.write(stream.read())

    return filename


def _get_latest_run(pipeline_id: str):
    # Call Bedrock API to get all runs of the training pipeline
    runs = pipeline_api.get_training_pipeline_runs(pipeline_id=pipeline_id)

    # Filter by creation time for the last successful run
    successful_runs = filter(lambda run: run.status == "Succeeded", runs)
    try:
        last_run = max(successful_runs, key=lambda run: run.updated_at)
    except ValueError as exc:
        raise Exception(
            f"No successful runs found for pipeline: {pipeline_id}"
        ) from exc
    return last_run


def download_artefact_from_latest_run(
    pipeline_id: str, output_filepath: Optional[str] = None
) -> Optional[str]:
    """
    Downloads the model artefact produced by the latest successful run of a given pipeline. The output
    filepath defaults to `/tmp/{pipeline_run_id}-artefact.zip`.

    :param pipeline_id: The public id of the pipeline to download from
    :type pipeline_id: str
    :param output_filepath: Output filename with path
    :type output_filepath: str
    :return: The file path that the artefact is saved to
    :rtype: str
    """

    last_run = _get_latest_run(pipeline_id)

    # Calls Bedrock API again to download artefact from the latest run
    filename = download_artefact_by_run_id(
        pipeline_id=pipeline_id,
        pipeline_run_id=last_run.entity_id,
        output_filepath=output_filepath
    )

    print(f"Downloaded artefact: {filename}")
    return filename


def download_and_unzip_latest_artefact(output_directory: Optional[str] = None) -> bool:
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

    pipeline = pipeline_api.get_training_pipeline_by_id(pipeline_id=pipeline_id)
    last_run = _get_latest_run(pipeline_id)

    # Extract the downloaded artefacts
    download_and_unzip_artefact(
        api_client=api_client,
        model_id=pipeline.model_id,
        model_artefact_id=last_run.artefact_id,
        output_dir=output_directory or f"/tmp/{pipeline_id}"
    )

    # List the contents for manual verification
    print(glob.glob(f"{output_directory}/*"))
    return True

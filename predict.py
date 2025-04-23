#!/usr/bin/env python

import os
import time
import wandb
import random
import shutil
import requests
import subprocess
import traceback
import sentry_sdk
from huggingface_hub import HfApi
from cog import BasePredictor, Input, Path, Secret

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

sentry_sdk.init(
    dsn="https://39edfbd91b390b746a8e2205f0a69627@o4506399435325440.ingest.us.sentry.io/4509157921718272",
    traces_sample_rate=1.0,
    environment="production",
)


class HuggingFaceDatasetValidator:
    @staticmethod
    def validate_dataset(repo_id: str) -> bool:
        """Validate if a Hugging Face dataset exists."""
        url = f"https://huggingface.co/api/datasets/{repo_id}"
        response = requests.get(url)
        return response.status_code == 200


class HuggingFaceTokenValidator:
    @staticmethod
    def has_write_access(hf_token: str, hf_model_name: str) -> bool:
        """Check if the HF token has write access by attempting to create a repo."""
        api = HfApi()
        try:
            api.create_repo(hf_model_name, private=False, exist_ok=True, token=hf_token)
            return True  # The token has write access
        except Exception as e:
            print(f"Write access check failed: {e}")
            return False  # The token does not have write access


class WandbValidator:
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate if a Weights & Biases API key is valid."""
        # Try to verify the key by running a simple wandb command
        if not api_key:
            return False
        try:
            # Just check if we can initialize wandb
            return wandb.login(key=api_key, relogin=True, verify=True)
        except Exception:
            return False
        finally:
            # Unset the API key
            if "WANDB_API_KEY" in os.environ:
                del os.environ["WANDB_API_KEY"]


def create_return_value(
    steps: int, dataset_repo_id: str, hf_model_name: str, wandb_run_url: str | None
) -> Path:
    """
    Create a README.md file with the model information and return the path to it.
    """
    # Create the README content
    readme_content = f"""
Thanks for using the phospho ACT pipeline! 🚀

Find your model on Hugging Face
- **Model**: [{hf_model_name}](https://huggingface.co/{hf_model_name})

Training parameters:

- **Dataset**: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})

- **Wandb run URL**: {wandb_run_url}

- **Training steps**: {steps}


More:

- 📖 **Get Started**: [docs.phospho.ai](https://docs.phospho.ai?utm_source=replicate)

- 🤖 **Get your robot**: [robots.phospho.ai](https://robots.phospho.ai?utm_source=replicate)
"""

    # Write the README file
    readme_path = Path("/tmp/outputs/train/README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    return readme_path


def upload_error_read_me_to_hf(
    hf_model_name: str,
    hf_token: str,
    error_traceback: str,
    dataset_repo_id: str,
    wandb_run_url: str | None,
    steps: int | None,
) -> Path:
    # Read me should explain why the pipeline failed
    readme_content = f"""
---
tags:
- phosphobot
- act
task_categories:
- robotics
---
# Act Model - phospho Training Pipeline

# Error Traceback

We faced an issue while training your model.

```
{error_traceback}
```

Training parameters:
- **Dataset**: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})
- **Wandb run URL**: {wandb_run_url}
- **Steps**: {steps}
- **Training steps**: {steps}

More:

- 📖 **Get Started**: [docs.phospho.ai](https://docs.phospho.ai?utm_source=replicate)

- 🤖 **Get your robot**: [robots.phospho.ai](https://robots.phospho.ai?utm_source=replicate)
"""
    # Write the README file
    readme_path = Path("/tmp/outputs/train/README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    # Upload the readme to Hugging Face
    api = HfApi()
    try:
        api.upload_file(
            repo_type="model",
            path_or_fileobj=str(readme_path.resolve()),
            path_in_repo="README.md",
            repo_id=hf_model_name,
            token=hf_token,
        )
    except Exception as e:
        print(f"Failed to upload readme to Hugging Face: {e}")
    return readme_path


def parse_hf_username_or_orgid(user_info: dict) -> str | None:
    """
    Extract the username or organization name from the user info dictionary.
    user_info = api.whoami(token=hf_token)
    """
    # Extract the username
    username = user_info.get("name", "Unknown")

    # If no fine grained permissions, return the username
    if user_info.get("auth", {}).get("accessToken", {}).get("role") == "write":
        return username

    # From now on, the token is assumed as fine-grained

    # Extract fine-grained permissions
    fine_grained_permissions = (
        user_info.get("auth", {}).get("accessToken", {}).get("fineGrained", {})
    )
    scoped_permissions = fine_grained_permissions.get("scoped", [])

    # Check if the token has write access to the user account
    for scope in scoped_permissions:
        entity = scope.get("entity", {})
        entity_type = entity.get("type")
        entity_name = entity.get("name")
        permissions = scope.get("permissions", [])

        # Check if the entity is the user and has write access
        if entity_type == "user" and "repo.write" in permissions:
            # Return the username
            return username

        # Check if the entity is an org and has write access
        if entity_type == "org" and "repo.write" in permissions:
            org_with_write_access = entity_name
            return org_with_write_access

    raise Exception(
        "No user or org with write access found. Wont be able to push to Hugging Face."
    )


def get_user_name_hf(token: str) -> str | None:
    api = HfApi()
    user_info = api.whoami(token=token)
    # Parse the user info to get the user name or the first org name
    user_name = parse_hf_username_or_orgid(user_info)
    return user_name


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Install lerobot and other dependencies"""
        pass

    def predict(
        self,
        dataset_repo_id: str = Input(
            description="Hugging Face dataset ID to train on, LeRobot format > v2.0 expected, i.e. 'LegrandFrederic/dual-setup'"
        ),
        hugging_face_token: Secret = Input(
            description="Hugging Face API token (used to upload your trained model to your HF profile), find yours here: https://huggingface.co/settings/token"
        ),
        hf_model_name: str = Input(
            description="Hugging Face model name, the name of the model to be created, i.e. 'trained-act-replicate'",
            default="",
        ),
        wandb_api_key: Secret = Input(
            description="Weights & Biases API key (optional, to track the online training), find yours here: https://wandb.ai/authorize",
            default="",
        ),
        wandb_project: str = Input(
            description="Weights & Biases project name",
            default="ACT-Replicate",
        ),
        job_name: str = Input(
            description="Name for this training job", default="replicate_training"
        ),
        steps: int = Input(
            description="Number of steps to train for", ge=1, le=200000, default=100000
        ),
        batch_size: int = Input(
            description="Batch size for training", ge=1, le=128, default=32
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        def capture_error(
            error: Exception,
            steps: int | None = None,
            wandb_run_url: str | None = None,
            upload_error_read_me_and_raise: bool = True,
            error_message_to_upload: str | None = None,
        ):
            error_traceback = traceback.format_exc()
            sentry_sdk.set_tag("hf_model_name", hf_model_name)
            sentry_sdk.set_tag("dataset_repo_id", dataset_repo_id)
            sentry_sdk.capture_exception(error)
            if upload_error_read_me_and_raise:
                upload_error_read_me_to_hf(
                    hf_model_name=hf_model_name,
                    hf_token=hf_token,
                    error_traceback=(
                        error_message_to_upload
                        if error_message_to_upload is not None
                        else error_traceback[-500:]
                    ),
                    dataset_repo_id=dataset_repo_id,
                    wandb_run_url=wandb_run_url,
                    steps=steps,
                )
                raise

        # Logging into HF
        hf_token: str = hugging_face_token.get_secret_value()
        hf_username = get_user_name_hf(hf_token)
        if hf_username is None:
            raise ValueError(
                "No username or organization with Write access found for token"
            )

        if hf_model_name == "":
            hf_model_name = (
                f"{hf_username}/gr00t-replicate-"
                + dataset_repo_id.replace("/", "-")
                + "-"
                + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))
            )
        else:
            # Check if the provided model name includes the username/organization
            if "/" not in hf_model_name:
                hf_model_name = f"{hf_username}/{hf_model_name}"

        # Check if we have the right to create a model
        print("Checking your HF token")
        if not HuggingFaceTokenValidator.has_write_access(
            hugging_face_token.get_secret_value(), hf_model_name
        ):
            raise ValueError(
                "The provided Hugging Face token does not have write access or your model name is already taken."
            )

        # Validate Hugging Face dataset
        print(f"Validating Hugging Face dataset: {dataset_repo_id}")
        if not HuggingFaceDatasetValidator.validate_dataset(dataset_repo_id):
            raise ValueError(
                f"Dataset '{dataset_repo_id}' not found on Hugging Face. Please check the dataset ID."
            )

        # Set up Weights & Biases if API key is provided
        wandb_enabled = wandb_api_key is not None
        wandb_run_url = None

        if wandb_enabled:
            wandb.login(key=wandb_api_key.get_secret_value())

        print("Weights & Biases enabled:", wandb_enabled)

        # Create output directory
        output_dir = Path(f"/tmp/outputs/train/{job_name}")

        try:
            api = HfApi()
            repo_url = api.create_repo(
                hf_model_name,
                private=False,
                exist_ok=True,
                token=hf_token,
            )
            print(
                f"Your model will be uploaded to: https://huggingface.co/{repo_url.repo_id}"
            )
            time.sleep(0.5)
        except Exception as e:
            capture_error(error=e)
        # Build the training command
        cmd = [
            "python",
            "-m",
            "lerobot.scripts.train",
            f"--dataset.repo_id={dataset_repo_id}",
            "--policy.type=act",
            f"--steps={steps}",
            f"--batch_size={batch_size}",
            f"--wandb.project={wandb_project}",
            "--policy.device=cuda",
            f"--output_dir={output_dir}",
            f"--wandb.enable={str(wandb_enabled).lower()}",
            f"--job_name={job_name}",
        ]

        print(f"Starting training with command: {' '.join(cmd)}")

        output_lines = []
        try:
            # Run the training command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    stripped_line = line.strip()
                    print(stripped_line)
                    output_lines.append(stripped_line)
            else:
                print("No output from training process")

            process.wait()

            if process.returncode != 0:
                error_output = "\n".join(output_lines[-10:])
                error_msg = f"Training process failed with exit code {process.returncode}:\n{error_output}"
                print(f"Training process failed with {error_msg}")
                raise RuntimeError(error_msg)
        except RuntimeError as e:
            capture_error(
                error=e,
                wandb_run_url=wandb_run_url,
                steps=steps,
                error_message_to_upload=str(e),
            )
        except Exception as e:
            capture_error(
                error=e,
                wandb_run_url=wandb_run_url,
                steps=steps,
            )
        # Upload to Hugging Face if token is provided

        try:
            huggingface_model_url = None
            if hugging_face_token.get_secret_value() and hf_model_name:
                print(f"Uploading model to Hugging Face as {hf_model_name}")
                # Set up token
                os.environ["HF_TOKEN"] = hugging_face_token.get_secret_value()

                # Upload using huggingface_hub
                from huggingface_hub import HfApi

                api = HfApi()

                files_directory = (
                    output_dir / "checkpoints" / "last" / "pretrained_model"
                )
                print(f"Uploading model files from {files_directory}")
                output_paths: list[Path] = []
                # Upload the model files
                for item in files_directory.glob("**/*"):
                    if item.is_file():
                        print(f"Uploading {item}")
                        api.upload_file(
                            repo_type="model",
                            path_or_fileobj=str(item.resolve()),
                            path_in_repo=item.name,
                            repo_id=repo_url.repo_id,
                            token=hugging_face_token.get_secret_value(),
                        )
                        output_paths.append(item)

                # Adding Readme
                readme_content = """
---
tags:
- phosphobot
- act
- replicate
task_categories:
- robotics                                               
---

# ACT Model - phospho Replication Pipeline

This model was trained using **phospho's Replicate pipeline** for **ACT models**, leveraging **LeRobot** for the training scripts.

🔗 **Explore on Replicate**: [Replicate](https://replicate.com/phospho-app/act-policy)

📖 **Get Started**: [docs.phospho.ai](https://docs.phospho.ai)

🤖 **Get your robot**: [robots.phospho.ai](https://robots.phospho.ai)
"""

                readme_path = files_directory / "README.md"
                with open(readme_path, "w") as f:
                    f.write(readme_content)

                api.upload_file(
                    repo_type="model",
                    path_or_fileobj=str(readme_path.resolve()),
                    path_in_repo="README.md",
                    repo_id=repo_url.repo_id,
                    token=hugging_face_token.get_secret_value(),
                )

                # Get the model URL
                huggingface_model_url = f"https://huggingface.co/{repo_url}"
                print(f"Model successfully uploaded to {huggingface_model_url}")
        except Exception as e:
            capture_error(error=e, wandb_run_url=wandb_run_url, steps=steps)

        print("Training completed successfully")
        print(f"Wandb run URL: {wandb_run_url}")
        print(f"Hugging Face model URL: {huggingface_model_url}")

        if os.path.exists("/root/.netrc"):
            os.remove("/root/.netrc")

        # Convert files in output_paths to zip
        archive = shutil.make_archive(
            base_name=files_directory, format="zip", root_dir=files_directory
        )

        return Path(archive)

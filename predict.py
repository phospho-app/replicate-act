#!/usr/bin/env python
# File: train.py

import os
import shutil
import requests
import subprocess
from huggingface_hub import HfApi
from cog import BasePredictor, Input, Path


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
        # Set the API key temporarily
        os.environ["WANDB_API_KEY"] = api_key

        # Try to verify the key by running a simple wandb command
        try:
            import wandb

            # Just check if we can initialize wandb
            _ = wandb.Api()
            return True
        except Exception:
            return False
        finally:
            # Unset the API key
            if "WANDB_API_KEY" in os.environ:
                del os.environ["WANDB_API_KEY"]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Install lerobot and other dependencies"""
        pass

    def predict(
        self,
        dataset_repo_id: str = Input(
            description="Hugging Face dataset ID to train on, LeRobot format > v2.0 expected, i.e. 'LegrandFrederic/dual-setup'"
        ),
        hf_token: str = Input(
            description="Hugging Face API token (used to upload your trained model to your HF profile), find yours here: https://huggingface.co/settings/token"
        ),
        hf_model_name: str = Input(
            description="Hugging Face model name, the name of the model to be created, i.e. 'trained-act-replicate'",
            default="trained-act-replicate",
        ),
        wandb_api_key: str = Input(
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
    ) -> Path:
        """Run a single prediction on the model"""
        # Logging into HF
        print("Logging into Hugging Face with the provided token")
        os.environ["HF_TOKEN"] = hf_token

        # Check if we have the right to create a model
        print("Checking your HF token")
        if not HuggingFaceTokenValidator.has_write_access(hf_token, hf_model_name):
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
        wandb_enabled = wandb_api_key != ""
        wandb_run_url = None

        if wandb_enabled:
            print("Validating Weights & Biases API key")
            if not WandbValidator.validate_api_key(wandb_api_key):
                raise ValueError(
                    "Invalid Weights & Biases API key. Please provide a valid API key."
                )

            # Set the API key
            os.environ["WANDB_API_KEY"] = wandb_api_key
            os.environ["WANDB_PROJECT"] = wandb_project

        # Create output directory
        output_dir = Path(f"/tmp/outputs/train/{job_name}")

        # Build the training command
        cmd = [
            "python",
            "-m",
            "lerobot.scripts.train",
            f"--dataset.repo_id={dataset_repo_id}",
            "--policy.type=act",
            f"--steps={steps}",
            f"--wandb.project={wandb_project}",
            "--policy.device=cuda",
            f"--output_dir={output_dir}",
            f"--wandb.enable={str(wandb_enabled).lower()}",
            f"--job_name={job_name}",
        ]

        print(f"Starting training with command: {' '.join(cmd)}")

        # Run the training command
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        # Stream the output
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            # Extract W&B run URL if present
            if wandb_enabled and "wandb:" in line and "View run at" in line:
                wandb_run_url = line.split("View run at")[-1].strip()

        # Wait for the process to complete
        process.wait()

        # Check if training was successful
        if process.returncode != 0:
            raise RuntimeError("Training failed. Check the logs for details.")

        # Upload to Hugging Face if token is provided
        huggingface_model_url = None
        if hf_token and hf_model_name:
            print(f"Uploading model to Hugging Face as {hf_model_name}")
            try:
                # Set up token
                os.environ["HF_TOKEN"] = hf_token

                # Upload using huggingface_hub
                from huggingface_hub import HfApi

                api = HfApi()
                repo_url = api.create_repo(hf_model_name, private=False, exist_ok=True)
                print(f"Created Hugging Face model repository: {repo_url.repo_id}")

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
                            token=hf_token,
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
                    token=hf_token,
                )

                # Get the model URL
                huggingface_model_url = f"https://huggingface.co/{repo_url}"
                print(f"Model successfully uploaded to {huggingface_model_url}")
            except Exception as e:
                print(f"Failed to upload model to Hugging Face: {e}")

        print("Training completed successfully")
        print(f"Wandb run URL: {wandb_run_url}")
        print(f"Hugging Face model URL: {huggingface_model_url}")

        # Convert files in output_paths to zip
        archive = shutil.make_archive(
            base_name=files_directory, format="zip", root_dir=files_directory
        )

        return Path(archive)

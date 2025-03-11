import os
import cog  # type: ignore
import subprocess
import requests
from pathlib import Path
import shutil
from typing import Optional


class HuggingFaceDatasetValidator:
    @staticmethod
    def validate_dataset(repo_id: str) -> bool:
        """Validate if a Hugging Face dataset exists."""
        url = f"https://huggingface.co/api/datasets/{repo_id}"
        response = requests.get(url)
        return response.status_code == 200


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


class TrainingOutput(cog.BaseModel):
    """Output from the training process"""

    model_path: str
    wandb_run_url: Optional[str]
    huggingface_model_url: Optional[str]


class Trainer(cog.Predictor):
    def setup(self):
        """Install lerobot and other dependencies"""
        # Install lerobot from GitHub
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/huggingface/lerobot.git",
                "/tmp/lerobot",
            ],
            check=True,
        )

        # Install lerobot
        subprocess.run(["pip", "install", "-e", "/tmp/lerobot"], check=True)

        # Install other possible dependencies
        subprocess.run(["pip", "install", "wandb", "huggingface_hub"], check=True)

    @cog.input(
        "dataset_repo_id",
        type=str,
        help="Hugging Face dataset repository ID (e.g., 'username/dataset_name')",
    )
    @cog.input(
        "job_name",
        type=str,
        default="replicate_training",
        help="Name for this training job",
    )
    @cog.input(
        "wandb_api_key",
        type=str,
        default="",
        help="Weights & Biases API key (optional)",
    )
    @cog.input(
        "wandb_project",
        type=str,
        default="ACT-Replicate",
        help="Weights & Biases project name",
    )
    @cog.input("epochs", type=int, default=100000, help="Number of training epochs")
    @cog.input("batch_size", type=int, default=32, help="Training batch size")
    @cog.input("learning_rate", type=float, default=5e-5, help="Learning rate")
    @cog.input(
        "hf_token",
        type=str,
        default="",
        help="Hugging Face API token for uploading the model (optional)",
    )
    @cog.input(
        "hf_model_name",
        type=str,
        default="",
        help="Name for the uploaded model on Hugging Face (required if hf_token is provided)",
    )
    def predict(
        self,
        dataset_repo_id: str,
        job_name: str,
        wandb_api_key: str,
        wandb_project: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        hf_token: str,
        hf_model_name: str,
    ) -> TrainingOutput:
        # Logging into HF
        if hf_token:
            print("Logging into Hugging Face with the provided token")
            os.environ["HF_TOKEN"] = hf_token
            subprocess.run(
                ["huggingface-cli", "login", "--api-key", hf_token], check=True
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
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build the training command
        cmd = [
            "python",
            "-m",
            "lerobot.scripts.train",
            f"--dataset.repo_id={dataset_repo_id}",
            "--policy.type=act",
            f"--output_dir={output_dir}",
            f"--job_name={job_name}",
            "--device=cuda",
            f"--wandb.enable={str(wandb_enabled).lower()}",
            f"--train.num_epochs={epochs}",
            f"--train.batch_size={batch_size}",
            f"--train.learning_rate={learning_rate}",
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

        # Copy the model to a more accessible location
        trained_model_path = Path(cog.Path("/outputs/model"))
        trained_model_path.mkdir(parents=True, exist_ok=True)

        # Copy the trained model files from the output directory
        for item in output_dir.glob("*"):
            if item.is_dir():
                shutil.copytree(item, trained_model_path / item.name)
            else:
                shutil.copy2(item, trained_model_path / item.name)

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
                api.create_repo(hf_model_name, private=False, exist_ok=True)

                # Upload the model files
                for item in trained_model_path.glob("**/*"):
                    if item.is_file():
                        relative_path = item.relative_to(trained_model_path)
                        api.upload_file(
                            path_or_fileobj=str(item),
                            path_in_repo=str(relative_path),
                            repo_id=hf_model_name,
                            token=hf_token,
                        )

                # Get the model URL
                huggingface_model_url = f"https://huggingface.co/{hf_model_name}"
                print(f"Model successfully uploaded to {huggingface_model_url}")
            except Exception as e:
                print(f"Failed to upload model to Hugging Face: {e}")

        return TrainingOutput(
            model_path="/outputs/model",
            wandb_run_url=wandb_run_url,
            huggingface_model_url=huggingface_model_url,
        )

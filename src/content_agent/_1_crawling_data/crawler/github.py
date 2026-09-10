import os
import shutil
import subprocess
import tempfile

from loguru import logger

from content_agent.crawling_data.documents.documents import RepositoryDocument

from .base import BaseCrawler


class GithubCrawler(BaseCrawler):
    model = RepositoryDocument # Specify the document model used to store the crawled repository.

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:

        # Check whether this repository has already been stored in the database.
        old_model = self.model.find(link=link)

        # Skip the crawling process if the repository already exists
        if old_model is not None:
            # Log the beginning of the crawling process.
            logger.info(f"Repository already exists in the database: {link}")

            return
        # Log the beginning of the crawling process.
        logger.info(f"Starting scrapping GitHub repository: {link}")


        # Extract the repository name from the GitHub URL.
        # Example: https://github.com/user/my-project/ -> my-project
        repo_name = link.rstrip("/").split("/")[-1]

        # Create a temporary directory where the repository will be cloned.
        local_temp = tempfile.mkdtemp()

        try:
            
            os.chdir(local_temp) # Change the current working directory to the temporary directory.
            subprocess.run(["git", "clone", link]) # Clone the GitHub repository using the git command.

            # Find the path of the cloned repository.
            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])  # noqa: PTH118

            # Store the contents of all accepted files.
            # The keys are file paths and the values are file contents.
            tree = {}

            # Recursively walk through all directories and files in the repository.
            for root, _, files in os.walk(repo_path):
                # Get the directory path relative to the repository root.
                dir = root.replace(repo_path, "").lstrip("/")

                # Skip ignored directories.
                if dir.startswith(self._ignore):
                    continue

                # Process each file in the current directory.
                for file in files:
                    # Skip files with ignored extensions.
                    if file.endswith(self._ignore):
                        continue

                    # Build the file path relative to the repository root.
                    file_path = os.path.join(dir, file)  # noqa: PTH118

                    # Open and read the file as text.
                    # errors="ignore" prevents decoding errors from stopping the crawler.
                    with open(os.path.join(root, file), "r", errors="ignore") as f:  # noqa: PTH123, PTH118
                        # Store the file content in the tree.
                        # Spaces are removed from the content.
                        tree[file_path] = f.read().replace(" ", "")

            # Get the user who triggered the crawling process.
            user = kwargs["user"]

            # Create a RepositoryDocument containing the crawled data.
            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()

        except Exception:
            raise
        finally:
            # Always remove the temporary directory,
            # whether the crawling process succeeds or fails.
            shutil.rmtree(local_temp)

        logger.info(f"Finished scrapping GitHub repository: {link}")

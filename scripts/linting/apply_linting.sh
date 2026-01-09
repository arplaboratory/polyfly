# apply linting with black with pyproject.toml
# Usage: ./apply_linting.sh
#!/bin/bash
# Check if black is installed
if ! command -v black &> /dev/null
then
    echo "black could not be found, please install it first."
    exit
fi
# Check if pyproject.toml exists
if [ ! -f "black.toml" ]; then
    echo "black.toml not found, please create it first."
    exit
fi
# Check if the current directory is a git repository
if [ ! -d ".git" ]; then
    echo "This script must be run in a git repository."
    exit
fi
# Check if there are any unstaged changes
if ! git diff-index --quiet HEAD --; then
    echo "There are unstaged changes in the repository. Please commit or stash them before running this script."
    exit
fi
# Check if there are any staged changes
if ! git diff-index --cached --quiet HEAD --; then
    echo "There are staged changes in the repository. Please commit or stash them before running this script."
    exit
fi

# apply linting 
echo "Applying linting with black..."
black --config black.toml .
if [ $? -ne 0 ]; then
    echo "Linting failed. Please fix the errors and try again."
    exit 1
fi
echo "Linting completed successfully."



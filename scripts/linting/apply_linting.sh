# apply linting with black with pyproject.toml
# Usage: ./apply_linting.sh
#!/bin/bash
# Check if black is installed
cd $POLYFLY_DIR 

if ! command -v black &> /dev/null
then
    echo "black could not be found, please install it first."
    exit
fi
# Check if pyproject.toml exists
if [ ! -f "scripts/linting/black.toml" ]; then
    echo "black.toml not found, please create it first."
    exit
fi
# Check if the current directory is a git repository
if [ ! -d ".git" ]; then
    echo "This script must be run in a git repository."
    exit
fi
# Check if there are any unstaged changes outside data directory
if ! git diff --quiet -- . ':!data'; then
    echo "There are unstaged changes outside the data directory."
    echo "Please commit or stash them before running this script."
    exit
fi

# apply linting 
echo "Applying linting with black..."
black --config scripts/linting/black.toml --exclude "$POLYFLY_DIR/data" .
if [ $? -ne 0 ]; then
    echo "Linting failed. Please fix the errors and try again."
    exit 1
fi
echo "Linting completed successfully."



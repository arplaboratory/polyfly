#!/usr/bin/env bash
# Development Docker run script - Mounts code as volume for live editing

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default image
IMAGE_TAG="${1:-poly-fly:latest}"
shift || true

# Get the workspace directory (parent of script directory)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}PolyFly Development Environment${NC}"
echo -e "Workspace: ${WORKSPACE_DIR}"
echo -e "Image: ${IMAGE_TAG}"
echo ""

# Enable X11 forwarding on host
xhost +local:docker || true

# Get user ID and group ID to run container as same user as host
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Create temp passwd/group files with current user
echo "$(id -un):x:$USER_ID:$GROUP_ID::/home/$(id -un):/bin/bash" > /tmp/passwd.docker
echo "$(id -gn):x:$GROUP_ID:" > /tmp/group.docker

# Run the container with automatic package installation
docker run --rm -it \
    --net=host \
    --name poly_fly_dev \
    --user $USER_ID:$GROUP_ID \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$WORKSPACE_DIR:/workspace:rw" \
    -v /tmp/passwd.docker:/etc/passwd:ro \
    -v /tmp/group.docker:/etc/group:ro \
    --workdir /workspace \
    "$IMAGE_TAG" \
    bash -c 'export PYTHONPATH=/workspace:$PYTHONPATH && export PS1="poly_fly@docker:\w$ " && /bin/bash -i'

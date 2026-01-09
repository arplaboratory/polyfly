# Development Dockerfile - Mounts code as volume for live editing
# CPU only version optimized for iterative development

# -------------------- Base Stage --------------------
FROM mambaorg/micromamba:latest
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libx11-6 tk tk-dev xterm && \
    apt-get clean

USER mambauser
WORKDIR /app

# Copy environment file
COPY conda/environment_polyfly.yml .

# Create Mamba Environment
RUN micromamba create -y -n poly_fly -f environment_polyfly.yml && \
    micromamba clean --all --force-pkgs-dirs -y

# Activate environment
ENV CONDA_DEFAULT_ENV=poly_fly
ENV PATH="/opt/conda/envs/poly_fly/bin:$PATH"

# Set the root directory environment variable
ENV POLYFLY_DIR=/workspace

# Don't install the package at build time - it will be mounted as a volume
# The dependencies are already installed via conda environment
USER root

# Install sudo for mambauser to install packages at runtime
RUN apt-get update && apt-get install -y sudo && \
    echo "mambauser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    apt-get clean

# Configure X11 forwarding support
ENV DISPLAY=${DISPLAY:-:0}
ENV QT_X11_NO_MITSHM=1

# Switch back to non-root user
USER mambauser

# Default to bash shell for interactive development
CMD ["/bin/bash"]

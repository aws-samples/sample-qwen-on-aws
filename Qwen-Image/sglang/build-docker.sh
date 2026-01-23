#!/bin/bash
# Optimized Docker build script with ECR push and cleanup
# Prevents system crashes with resource limits and saves disk space after build

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from Jupyter notebook
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-459006231907}"
ECR_REPO_NAME="${ECR_REPO_NAME:-qwen-image-sglang}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

# Resource limits to prevent system crashes
# Note: --cpus flag not supported in docker build, only memory limits apply
MEMORY_LIMIT="${MEMORY_LIMIT:-8g}"
MEMORY_SWAP_LIMIT="${MEMORY_SWAP_LIMIT:-16g}"

# Cleanup configuration
CLEANUP_ENABLED="${CLEANUP_ENABLED:-true}"
KEEP_LOCAL_IMAGE="${KEEP_LOCAL_IMAGE:-false}"

# Derived values
ECR_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
LOCAL_IMAGE_NAME="${ECR_REPO_NAME}:${IMAGE_TAG}"
BUILD_LOG="build-$(date +%Y%m%d-%H%M%S).log"

echo -e "${BLUE}=========================================="
echo "Docker Build, Push to ECR, and Cleanup"
echo -e "==========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  AWS Region: $AWS_REGION"
echo "  AWS Account: $AWS_ACCOUNT_ID"
echo "  ECR Repo: $ECR_REPO_NAME"
echo "  Image Tag: $IMAGE_TAG"
echo "  Dockerfile: $DOCKERFILE"
echo "  ECR URI: $ECR_IMAGE_URI"
echo ""
echo -e "${YELLOW}Resource Limits:${NC}"
echo "  Memory: $MEMORY_LIMIT"
echo "  Memory+Swap: $MEMORY_SWAP_LIMIT"
echo ""
echo -e "${YELLOW}Cleanup:${NC}"
echo "  Enabled: $CLEANUP_ENABLED"
echo "  Keep local image: $KEEP_LOCAL_IMAGE"
echo ""

# Function to display disk usage
show_disk_usage() {
    echo -e "${BLUE}Current disk usage:${NC}"
    df -h . | tail -1
    echo ""
}

# Function to cleanup Docker resources
cleanup_docker() {
    echo -e "${YELLOW}=========================================="
    echo "Cleaning up Docker resources..."
    echo -e "==========================================${NC}"

    # Remove dangling images
    echo -e "${BLUE}Removing dangling images...${NC}"
    docker image prune -f || true

    # Remove build cache
    echo -e "${BLUE}Removing build cache...${NC}"
    docker builder prune -f || true

    # Remove stopped containers
    echo -e "${BLUE}Removing stopped containers...${NC}"
    docker container prune -f || true

    # Optionally remove the local image if not needed
    if [ "$KEEP_LOCAL_IMAGE" = "false" ]; then
        echo -e "${BLUE}Removing local image: $LOCAL_IMAGE_NAME${NC}"
        docker rmi "$LOCAL_IMAGE_NAME" 2>/dev/null || true
        docker rmi "$ECR_IMAGE_URI" 2>/dev/null || true
    else
        echo -e "${GREEN}Keeping local image: $LOCAL_IMAGE_NAME${NC}"
    fi

    # Clean up old build logs (keep last 5)
    echo -e "${BLUE}Cleaning up old build logs...${NC}"
    ls -t build-*.log 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true

    echo -e "${GREEN}Cleanup completed!${NC}"
    echo ""
}

# Show initial disk usage
echo -e "${BLUE}=========================================="
echo "Initial State"
echo -e "==========================================${NC}"
show_disk_usage

# Step 1: Create ECR repository if it doesn't exist
echo -e "${BLUE}=========================================="
echo "Step 1: Creating ECR Repository"
echo -e "==========================================${NC}"
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" &>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION"
echo -e "${GREEN}ECR repository ready: $ECR_REPO_NAME${NC}"
echo ""

# Step 2: Authenticate Docker with ECR
echo -e "${BLUE}=========================================="
echo "Step 2: Authenticating with ECR"
echo -e "==========================================${NC}"
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo -e "${GREEN}Docker authenticated with ECR${NC}"
echo ""

# Step 3: Build Docker image with resource limits
echo -e "${BLUE}=========================================="
echo "Step 3: Building Docker Image"
echo -e "==========================================${NC}"
echo "This may take 5-10 minutes..."
echo "Build log: $BUILD_LOG"
echo ""

# Add timeout to prevent runaway builds (20 minutes should be plenty for wheel install)
timeout 20m docker build \
  --memory="$MEMORY_LIMIT" \
  --memory-swap="$MEMORY_SWAP_LIMIT" \
  --network=host \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "${LOCAL_IMAGE_NAME}" \
  -f "$DOCKERFILE" \
  . 2>&1 | tee "$BUILD_LOG"

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""

# Step 4: Tag image for ECR
echo -e "${BLUE}=========================================="
echo "Step 4: Tagging Image for ECR"
echo -e "==========================================${NC}"
docker tag "${LOCAL_IMAGE_NAME}" "${ECR_IMAGE_URI}"
echo -e "${GREEN}Tagged: $ECR_IMAGE_URI${NC}"
echo ""

# Step 5: Push to ECR
echo -e "${BLUE}=========================================="
echo "Step 5: Pushing Image to ECR"
echo -e "==========================================${NC}"
echo "This may take several minutes depending on image size..."
echo ""
docker push "${ECR_IMAGE_URI}"
echo ""
echo -e "${GREEN}Image pushed successfully!${NC}"
echo -e "${GREEN}ECR URI: $ECR_IMAGE_URI${NC}"
echo ""

# Step 6: Cleanup (if enabled)
if [ "$CLEANUP_ENABLED" = "true" ]; then
    cleanup_docker
    show_disk_usage
else
    echo -e "${YELLOW}Cleanup disabled. Skipping cleanup steps.${NC}"
    echo ""
fi

# Summary
echo -e "${GREEN}=========================================="
echo "SUCCESS! All steps completed"
echo -e "==========================================${NC}"
echo ""
echo -e "${GREEN}Image Details:${NC}"
echo "  Local: $LOCAL_IMAGE_NAME"
echo "  ECR: $ECR_IMAGE_URI"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Use the ECR URI in your SageMaker notebook"
echo "  2. Deploy to SageMaker endpoint"
echo ""
echo -e "${YELLOW}To manually cleanup later:${NC}"
echo "  docker image prune -a -f"
echo "  docker builder prune -a -f"
echo ""

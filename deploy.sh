#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# nulltracer deploy script
#
# GPU-accelerated black hole ray tracer — deployment automation
# Handles container builds, static file deployment, health checks, and rollback.
#
# Usage: ./deploy.sh [command]
#   Commands:
#     all      Full deploy: build + static + health check (default)
#     build    Build and restart only the renderer container
#     static   Deploy only static files to /srv/nulltracer
#     status   Show container status and health
#     logs     Tail renderer container logs
#     rollback Restore previous Docker image
#     help     Show this help message
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
SERVER_DIR="${SCRIPT_DIR}/server"
STATIC_DEST="/srv/nulltracer"
LOCK_FILE="/tmp/nulltracer-deploy.lock"

CONTAINER_NAME="nulltracer-renderer"
IMAGE_NAME="nulltracer-renderer"
HEALTH_URL="http://localhost:8420/health"
HEALTH_TIMEOUT=150  # seconds to wait for health check (CuPy kernel compilation can take 120s + 30s margin)
HEALTH_INTERVAL=3   # seconds between health check polls

# ── Colors & Output ─────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[OK]${NC}      $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC}   $*" >&2; }

# ── Timing ───────────────────────────────────────────────────────────────────

timer_start() { STEP_START=$(date +%s); }
timer_end()   {
    local elapsed=$(( $(date +%s) - STEP_START ))
    info "Step completed in ${elapsed}s"
}

# ── Lock File ────────────────────────────────────────────────────────────────

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local lock_pid
        lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            error "Another deploy is already running (PID ${lock_pid})"
            exit 1
        else
            warn "Stale lock file found, removing"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
    trap release_lock EXIT
}

release_lock() {
    rm -f "$LOCK_FILE"
}

# ── Pre-flight Checks ───────────────────────────────────────────────────────

preflight() {
    info "Running pre-flight checks..."

    # Docker daemon
    if ! docker info &>/dev/null; then
        error "Docker is not running or not accessible"
        exit 1
    fi
    success "Docker daemon is running"

    # docker compose
    if docker compose version &>/dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        error "Neither 'docker compose' nor 'docker-compose' found"
        exit 1
    fi
    success "Compose available: ${COMPOSE_CMD}"

    # curl (needed for health checks)
    if ! command -v curl &>/dev/null; then
        error "curl is required for health checks but is not installed"
        exit 1
    fi
    success "curl available"

    # NVIDIA runtime
    if ! docker info 2>/dev/null | grep -qi "nvidia"; then
        warn "NVIDIA runtime not detected in 'docker info' — GPU passthrough may fail"
    else
        success "NVIDIA runtime available"
    fi

    # Compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "Compose file not found: ${COMPOSE_FILE}"
        exit 1
    fi
    success "Compose file found"
}

# ── Health Check ─────────────────────────────────────────────────────────────

health_check() {
    info "Waiting for health check at ${HEALTH_URL} (timeout: ${HEALTH_TIMEOUT}s)..."
    local elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -sf "$HEALTH_URL" &>/dev/null; then
            success "Health check passed after ${elapsed}s"
            return 0
        fi
        sleep "$HEALTH_INTERVAL"
        elapsed=$(( elapsed + HEALTH_INTERVAL ))
    done
    error "Health check failed after ${HEALTH_TIMEOUT}s"
    return 1
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_build() {
    info "${BOLD}Building renderer container...${NC}"
    timer_start

    # Step 1: Build the new image (don't stop anything yet)
    info "Building new image..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build renderer
    success "Image built successfully"

    # Step 2: Tag current image as :previous for rollback
    local current_image
    current_image=$(docker inspect --format='{{.Image}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
    if [ -n "$current_image" ]; then
        info "Tagging current image as ${IMAGE_NAME}:previous for rollback..."
        docker tag "$current_image" "${IMAGE_NAME}:previous" 2>/dev/null || \
            warn "Could not tag current image for rollback (container may not exist yet)"
    else
        warn "No running container found — skipping rollback tag (first deploy?)"
    fi

    # Step 3: Stop and replace with new image
    info "Replacing container..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d --force-recreate renderer
    success "Container replaced"

    # Step 4: Health check
    if ! health_check; then
        error "Deploy failed — health check did not pass"
        if docker image inspect "${IMAGE_NAME}:previous" &>/dev/null; then
            warn "Attempting automatic rollback..."
            cmd_rollback
        else
            error "No previous image available for rollback"
        fi
        exit 1
    fi

    timer_end
    success "${BOLD}Renderer deploy complete${NC}"
}

cmd_static() {
    info "${BOLD}Deploying static files to ${STATIC_DEST}...${NC}"
    timer_start

    # Ensure destination exists
    if [ ! -d "$STATIC_DEST" ]; then
        info "Creating ${STATIC_DEST}..."
        sudo mkdir -p "$STATIC_DEST"
    fi

    # Files and directories to deploy
    local -a files=(
        "index.html"
        "bench.html"
        "styles.css"
        "nulltracer-icon.png"
        "nulltracer-icon.svg"
    )
    local -a dirs=(
        "js"
    )

    # Use rsync if available, fall back to cp
    if command -v rsync &>/dev/null; then
        info "Using rsync for file deployment..."
        for f in "${files[@]}"; do
            if [ -f "${SCRIPT_DIR}/${f}" ]; then
                sudo rsync -a "${SCRIPT_DIR}/${f}" "${STATIC_DEST}/${f}"
            else
                warn "File not found: ${f} — skipping"
            fi
        done
        for d in "${dirs[@]}"; do
            if [ -d "${SCRIPT_DIR}/${d}" ]; then
                sudo rsync -a --delete "${SCRIPT_DIR}/${d}/" "${STATIC_DEST}/${d}/"
            else
                warn "Directory not found: ${d} — skipping"
            fi
        done
    else
        info "rsync not found, using cp..."
        for f in "${files[@]}"; do
            if [ -f "${SCRIPT_DIR}/${f}" ]; then
                sudo cp -f "${SCRIPT_DIR}/${f}" "${STATIC_DEST}/${f}"
            else
                warn "File not found: ${f} — skipping"
            fi
        done
        for d in "${dirs[@]}"; do
            if [ -d "${SCRIPT_DIR}/${d}" ]; then
                sudo cp -rf "${SCRIPT_DIR}/${d}" "${STATIC_DEST}/"
            else
                warn "Directory not found: ${d} — skipping"
            fi
        done
    fi

    success "Static files deployed"
    timer_end
}

cmd_status() {
    echo -e "${BOLD}Container Status:${NC}"
    echo "────────────────────────────────────────"
    if docker inspect "$CONTAINER_NAME" &>/dev/null; then
        local state health
        state=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME")
        health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no healthcheck{{end}}' "$CONTAINER_NAME")
        local started
        started=$(docker inspect --format='{{.State.StartedAt}}' "$CONTAINER_NAME")
        echo -e "  Container:  ${CONTAINER_NAME}"
        echo -e "  State:      ${state}"
        echo -e "  Health:     ${health}"
        echo -e "  Started:    ${started}"
        echo ""
        echo -e "${BOLD}Port Bindings:${NC}"
        docker port "$CONTAINER_NAME" 2>/dev/null || echo "  (none)"
        echo ""
        echo -e "${BOLD}Image:${NC}"
        docker inspect --format='  ID:   {{.Image}}' "$CONTAINER_NAME"
        echo ""
        # Check if previous image exists for rollback
        if docker image inspect "${IMAGE_NAME}:previous" &>/dev/null; then
            success "Rollback image available (${IMAGE_NAME}:previous)"
        else
            info "No rollback image tagged"
        fi
    else
        warn "Container '${CONTAINER_NAME}' not found"
    fi
}

cmd_logs() {
    info "Tailing logs for ${CONTAINER_NAME}..."
    docker logs -f "$CONTAINER_NAME"
}

cmd_rollback() {
    info "${BOLD}Rolling back to previous image...${NC}"
    timer_start

    if ! docker image inspect "${IMAGE_NAME}:previous" &>/dev/null; then
        error "No previous image found (${IMAGE_NAME}:previous)"
        error "Cannot rollback — no prior deploy was tagged"
        exit 1
    fi

    # Tag the previous image as the current build target
    info "Restoring previous image..."
    docker tag "${IMAGE_NAME}:previous" "${IMAGE_NAME}:latest" 2>/dev/null || true

    # Recreate the container with the restored image (--no-build prevents rebuild from source)
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d --no-build --force-recreate renderer
    success "Container rolled back"

    if health_check; then
        success "${BOLD}Rollback complete — service is healthy${NC}"
    else
        error "Rollback deployed but health check failed — manual intervention required"
        exit 1
    fi

    timer_end
}

cmd_all() {
    info "${BOLD}Starting full deployment...${NC}"
    local deploy_start
    deploy_start=$(date +%s)

    cmd_build
    cmd_static

    local total_elapsed=$(( $(date +%s) - deploy_start ))
    echo ""
    success "${BOLD}Full deployment complete in ${total_elapsed}s${NC}"
}

cmd_help() {
    echo -e "${BOLD}nulltracer deploy${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all        Full deploy: build + static + health check (default)"
    echo "  build      Build and restart only the renderer container"
    echo "  static     Deploy only static files to ${STATIC_DEST}"
    echo "  status     Show container status and health"
    echo "  logs       Tail renderer container logs"
    echo "  rollback   Restore previous Docker image"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Full deploy (same as 'all')"
    echo "  $0 build        # Rebuild and restart renderer only"
    echo "  $0 static       # Update static files only"
    echo "  $0 rollback     # Revert to previous image"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    local command="${1:-all}"

    case "$command" in
        help|--help|-h)
            cmd_help
            exit 0
            ;;
    esac

    # Commands that don't need full preflight or locking
    case "$command" in
        status)
            cmd_status
            exit 0
            ;;
        logs)
            cmd_logs
            exit 0
            ;;
    esac

    # All other commands need preflight and locking
    acquire_lock
    preflight

    case "$command" in
        all)      cmd_all ;;
        build)    cmd_build ;;
        static)   cmd_static ;;
        rollback) cmd_rollback ;;
        *)
            error "Unknown command: ${command}"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"

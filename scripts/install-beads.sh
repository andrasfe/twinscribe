#!/usr/bin/env bash
# Install Beads - Git-backed issue tracker for AI agents
# See: https://github.com/steveyegge/beads

set -euo pipefail

BEADS_VERSION="${BEADS_VERSION:-latest}"

echo "Installing Beads (git-backed issue tracker)..."

# Detect OS and architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$ARCH" in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Install based on available package managers
install_via_npm() {
    if command -v npm &> /dev/null; then
        echo "Installing via npm..."
        npm install -g beads-cli
        return 0
    fi
    return 1
}

install_via_homebrew() {
    if command -v brew &> /dev/null; then
        echo "Installing via Homebrew..."
        brew install steveyegge/tap/beads
        return 0
    fi
    return 1
}

install_via_go() {
    if command -v go &> /dev/null; then
        echo "Installing via Go..."
        go install github.com/steveyegge/beads/cmd/bd@${BEADS_VERSION}
        return 0
    fi
    return 1
}

install_via_binary() {
    echo "Downloading pre-built binary..."

    INSTALL_DIR="${BEADS_INSTALL_DIR:-/usr/local/bin}"
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    # Determine download URL
    if [ "$BEADS_VERSION" = "latest" ]; then
        DOWNLOAD_URL="https://github.com/steveyegge/beads/releases/latest/download/bd-${OS}-${ARCH}"
    else
        DOWNLOAD_URL="https://github.com/steveyegge/beads/releases/download/${BEADS_VERSION}/bd-${OS}-${ARCH}"
    fi

    echo "Downloading from: $DOWNLOAD_URL"

    if command -v curl &> /dev/null; then
        curl -fsSL "$DOWNLOAD_URL" -o "$TEMP_DIR/bd"
    elif command -v wget &> /dev/null; then
        wget -q "$DOWNLOAD_URL" -O "$TEMP_DIR/bd"
    else
        echo "Error: Neither curl nor wget found"
        return 1
    fi

    chmod +x "$TEMP_DIR/bd"

    # Install to target directory
    if [ -w "$INSTALL_DIR" ]; then
        mv "$TEMP_DIR/bd" "$INSTALL_DIR/bd"
    else
        echo "Installing to $INSTALL_DIR requires sudo..."
        sudo mv "$TEMP_DIR/bd" "$INSTALL_DIR/bd"
    fi

    echo "Installed bd to $INSTALL_DIR/bd"
    return 0
}

# Try installation methods in order of preference
if install_via_homebrew; then
    :
elif install_via_npm; then
    :
elif install_via_go; then
    :
elif install_via_binary; then
    :
else
    echo "Failed to install Beads. Please install manually:"
    echo "  - npm: npm install -g beads-cli"
    echo "  - Homebrew: brew install steveyegge/tap/beads"
    echo "  - Go: go install github.com/steveyegge/beads/cmd/bd@latest"
    echo "  - Manual: https://github.com/steveyegge/beads/releases"
    exit 1
fi

# Verify installation
if command -v bd &> /dev/null; then
    echo ""
    echo "Beads installed successfully!"
    bd --version 2>/dev/null || echo "bd is installed"
    echo ""
    echo "Quick start:"
    echo "  bd init          # Initialize Beads in current repo"
    echo "  bd ready         # Show tasks with no blocking dependencies"
    echo "  bd create 'Task' # Create a new task"
    echo "  bd show <id>     # View task details"
    echo ""
    echo "For more info: https://github.com/steveyegge/beads"
else
    echo "Warning: bd command not found in PATH"
    echo "You may need to add the install directory to your PATH"
    exit 1
fi

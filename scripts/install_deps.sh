#!/bin/bash
# PQC Scheduler - Dependency Installation Script
# Copyright (c) 2025 Dyber, Inc. All Rights Reserved.

set -e

echo "=============================================="
echo "PQC Scheduler Dependency Installation"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root for system packages
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}Note: Some operations may require sudo${NC}"
    fi
}

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        echo -e "${RED}Unsupported OS${NC}"
        exit 1
    fi
    echo "Detected OS: $OS $VERSION"
}

# Install system dependencies
install_system_deps() {
    echo ""
    echo "Installing system dependencies..."
    
    case $OS in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                ninja-build \
                git \
                pkg-config \
                libssl-dev \
                libclang-dev \
                llvm-dev \
                curl \
                wget \
                redis-server \
                python3 \
                python3-pip \
                python3-venv
            ;;
        fedora|rhel|centos)
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                cmake \
                ninja-build \
                git \
                pkgconfig \
                openssl-devel \
                clang-devel \
                llvm-devel \
                curl \
                wget \
                redis \
                python3 \
                python3-pip
            ;;
        arch|manjaro)
            sudo pacman -Sy --noconfirm \
                base-devel \
                cmake \
                ninja \
                git \
                pkg-config \
                openssl \
                clang \
                llvm \
                curl \
                wget \
                redis \
                python \
                python-pip
            ;;
        *)
            echo -e "${RED}Unsupported distribution: $OS${NC}"
            echo "Please install the following manually:"
            echo "  - build-essential/gcc"
            echo "  - cmake"
            echo "  - openssl development libraries"
            echo "  - clang/llvm"
            echo "  - redis"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}System dependencies installed${NC}"
}

# Install Rust
install_rust() {
    echo ""
    echo "Checking Rust installation..."
    
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        echo "Rust $RUST_VERSION already installed"
        
        # Check minimum version
        MIN_VERSION="1.75.0"
        if [ "$(printf '%s\n' "$MIN_VERSION" "$RUST_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
            echo -e "${YELLOW}Rust version $RUST_VERSION is below minimum $MIN_VERSION${NC}"
            echo "Updating Rust..."
            rustup update stable
        fi
    else
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    
    # Add required components
    rustup component add rustfmt clippy
    
    echo -e "${GREEN}Rust ready${NC}"
}

# Install liboqs
install_liboqs() {
    echo ""
    echo "Installing liboqs..."
    
    LIBOQS_VERSION="0.10.0"
    LIBOQS_COMMIT="a5e4814"
    LIBOQS_DIR="/tmp/liboqs-build"
    
    # Check if already installed
    if [ -f /usr/local/lib/liboqs.so ] || [ -f /usr/local/lib/liboqs.a ]; then
        echo "liboqs already installed, checking version..."
        # For simplicity, we'll reinstall
    fi
    
    # Clean previous build
    rm -rf "$LIBOQS_DIR"
    mkdir -p "$LIBOQS_DIR"
    cd "$LIBOQS_DIR"
    
    # Clone and checkout specific version
    git clone https://github.com/open-quantum-safe/liboqs.git
    cd liboqs
    git checkout "$LIBOQS_COMMIT"
    
    # Build
    mkdir build && cd build
    cmake -GNinja \
        -DOQS_USE_AVX2=ON \
        -DOQS_USE_AVX512=OFF \
        -DOQS_BUILD_ONLY_LIB=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        ..
    
    ninja
    sudo ninja install
    
    # Update library cache
    sudo ldconfig
    
    cd /
    rm -rf "$LIBOQS_DIR"
    
    echo -e "${GREEN}liboqs $LIBOQS_VERSION installed${NC}"
}

# Install oqs-provider for OpenSSL
install_oqs_provider() {
    echo ""
    echo "Installing oqs-provider..."
    
    OQS_PROVIDER_VERSION="0.5.3"
    OQS_PROVIDER_DIR="/tmp/oqs-provider-build"
    
    rm -rf "$OQS_PROVIDER_DIR"
    mkdir -p "$OQS_PROVIDER_DIR"
    cd "$OQS_PROVIDER_DIR"
    
    git clone https://github.com/open-quantum-safe/oqs-provider.git
    cd oqs-provider
    git checkout "$OQS_PROVIDER_VERSION"
    
    mkdir build && cd build
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        ..
    
    ninja
    sudo ninja install
    
    cd /
    rm -rf "$OQS_PROVIDER_DIR"
    
    echo -e "${GREEN}oqs-provider $OQS_PROVIDER_VERSION installed${NC}"
}

# Install Python dependencies for analysis
install_python_deps() {
    echo ""
    echo "Installing Python dependencies..."
    
    pip3 install --user \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scipy \
        jupyter \
        notebook
    
    echo -e "${GREEN}Python dependencies installed${NC}"
}

# Start Redis
start_redis() {
    echo ""
    echo "Starting Redis..."
    
    if systemctl is-active --quiet redis-server 2>/dev/null || \
       systemctl is-active --quiet redis 2>/dev/null; then
        echo "Redis already running"
    else
        sudo systemctl start redis-server 2>/dev/null || \
        sudo systemctl start redis 2>/dev/null || \
        echo -e "${YELLOW}Could not start Redis automatically. Please start manually.${NC}"
    fi
}

# Verify installation
verify_installation() {
    echo ""
    echo "=============================================="
    echo "Verifying installation..."
    echo "=============================================="
    
    ERRORS=0
    
    # Check Rust
    if command -v rustc &> /dev/null; then
        echo -e "${GREEN}✓ Rust: $(rustc --version)${NC}"
    else
        echo -e "${RED}✗ Rust not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check Cargo
    if command -v cargo &> /dev/null; then
        echo -e "${GREEN}✓ Cargo: $(cargo --version)${NC}"
    else
        echo -e "${RED}✗ Cargo not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check liboqs
    if [ -f /usr/local/lib/liboqs.so ] || [ -f /usr/local/lib/liboqs.a ]; then
        echo -e "${GREEN}✓ liboqs installed${NC}"
    else
        echo -e "${RED}✗ liboqs not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check OpenSSL
    if command -v openssl &> /dev/null; then
        echo -e "${GREEN}✓ OpenSSL: $(openssl version)${NC}"
    else
        echo -e "${RED}✗ OpenSSL not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        echo -e "${GREEN}✓ Redis: $(redis-cli --version)${NC}"
    else
        echo -e "${YELLOW}! Redis CLI not found (optional)${NC}"
    fi
    
    echo ""
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}All dependencies installed successfully!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. cd pqc-scheduler"
        echo "  2. cargo build --release"
        echo "  3. cargo test"
        echo "  4. cargo run --release -- --help"
    else
        echo -e "${RED}$ERRORS error(s) during installation${NC}"
        exit 1
    fi
}

# Main installation flow
main() {
    check_sudo
    detect_os
    install_system_deps
    install_rust
    install_liboqs
    install_oqs_provider
    install_python_deps
    start_redis
    verify_installation
}

# Run main
main "$@"

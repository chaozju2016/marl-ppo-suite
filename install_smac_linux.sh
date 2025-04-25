#!/bin/bash

# Install StarCraft II and SMACv2 maps for Linux
# This script downloads and installs StarCraft II and the SMACv2 maps to the correct location

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default StarCraft II installation path for Linux
SC2PATH="$HOME/StarCraftII"

# Print header
echo -e "${GREEN}=== StarCraft II and SMACv2 Maps Installer for Linux ===${NC}"
echo ""

# Check if StarCraft II is installed
if [ ! -d "$SC2PATH" ]; then
    echo -e "${YELLOW}StarCraft II is not installed. Installing now...${NC}"

    # Install required packages
    echo -e "${YELLOW}Checking for required packages...${NC}"
    if ! command -v unzip &> /dev/null || ! command -v wget &> /dev/null; then
        echo -e "${YELLOW}Installing required packages...${NC}"
        sudo apt-get update
        sudo apt-get install -y unzip wget
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error: Failed to install required packages.${NC}"
            echo "Please install unzip and wget manually and try again."
            exit 1
        fi
    fi

    # Create a temporary directory for downloads
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Download StarCraft II
    echo -e "${YELLOW}Downloading StarCraft II (4.10)...${NC}"
    echo -e "${YELLOW}This may take a while depending on your internet connection...${NC}"
    wget -q --show-progress http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to download StarCraft II.${NC}"
        echo "Please check your internet connection and try again."
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    # Extract StarCraft II
    echo -e "${YELLOW}Extracting StarCraft II...${NC}"
    echo -e "${YELLOW}This may take a while...${NC}"
    unzip -P iagreetotheeula SC2.4.10.zip -d "$HOME"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to extract StarCraft II.${NC}"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    # Clean up
    rm -rf "$TEMP_DIR"

    echo -e "${GREEN}StarCraft II successfully installed to $SC2PATH!${NC}"
fi

echo -e "${GREEN}StarCraft II installation path set to:${NC} $SC2PATH"

# Install required packages
echo -e "${YELLOW}Checking for required packages...${NC}"
if ! command -v unzip &> /dev/null; then
    echo -e "${YELLOW}Installing unzip...${NC}"
    sudo apt-get update
    sudo apt-get install -y unzip
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install unzip.${NC}"
        echo "Please install unzip manually and try again."
        exit 1
    fi
fi

# Create Maps directory if it doesn't exist
MAP_DIR="$SC2PATH/Maps"
if [ ! -d "$MAP_DIR" ]; then
    echo -e "${YELLOW}Creating Maps directory...${NC}"
    mkdir -p "$MAP_DIR"
fi

# Create SMAC_Maps directory if it doesn't exist
SMAC_MAPS_DIR="$MAP_DIR/SMAC_Maps"
if [ ! -d "$SMAC_MAPS_DIR" ]; then
    echo -e "${YELLOW}Creating SMAC_Maps directory...${NC}"
    mkdir -p "$SMAC_MAPS_DIR"
fi

# Create a temporary directory for downloads
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download SMACv2 maps
echo -e "${YELLOW}Downloading SMACv2 maps...${NC}"
wget -q https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to download SMACv2 maps.${NC}"
    echo "Please check your internet connection and try again."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Extract maps
echo -e "${YELLOW}Extracting maps...${NC}"
unzip -o SMAC_Maps.zip

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to extract maps.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copy maps to StarCraft II directory
echo -e "${YELLOW}Installing maps to $SMAC_MAPS_DIR...${NC}"
cp -f *.SC2Map "$SMAC_MAPS_DIR/"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to copy maps to $SMAC_MAPS_DIR.${NC}"
    echo "You might need to run this script with sudo."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Clean up
rm -rf "$TEMP_DIR"

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
if [ -f "$SMAC_MAPS_DIR/32x32_flat.SC2Map" ]; then
    echo -e "${GREEN}SMACv2 maps successfully installed!${NC}"
    echo "Maps are located at: $SMAC_MAPS_DIR"
    echo "You should now be able to run SMACv2 environments."
else
    echo -e "${RED}Error: Installation verification failed.${NC}"
    echo "The maps may not have been installed correctly."
    exit 1
fi

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"

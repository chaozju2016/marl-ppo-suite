#!/bin/bash

# Install SMACv2 maps for macOS
# This script downloads and installs the SMACv2 maps to the correct location

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default StarCraft II installation path for macOS
SC2PATH="/Applications/StarCraft II"

# Print header
echo -e "${GREEN}=== SMACv2 Maps Installer for macOS ===${NC}"
echo ""

# Check if StarCraft II is installed
if [ ! -d "$SC2PATH" ]; then
    echo -e "${YELLOW}Warning: StarCraft II installation not found at $SC2PATH${NC}"
    echo "Would you like to:"
    echo "1) Specify a different StarCraft II installation path"
    echo "2) Get instructions for installing StarCraft II"
    echo "3) Install maps anyway to $SC2PATH (will create directories)"
    echo "4) Exit"
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            read -p "Enter the path to your StarCraft II installation: " SC2PATH
            if [ ! -d "$SC2PATH" ]; then
                echo -e "${RED}Error: Directory $SC2PATH does not exist.${NC}"
                exit 1
            fi
            ;;
        2)
            echo -e "${YELLOW}Instructions for installing StarCraft II on macOS:${NC}"
            echo ""
            echo "1. Visit the Blizzard website: https://starcraft2.com/"
            echo "2. Click on 'Play Free Now' or 'Download'"
            echo "3. Sign in with your Blizzard account or create a new one"
            echo "4. Download the Battle.net installer"
            echo "5. Install Battle.net"
            echo "6. Open Battle.net and install StarCraft II"
            echo "7. Once installed, run StarCraft II at least once to complete setup"
            echo "8. After installation, run this script again to install the SMACv2 maps"
            echo ""
            echo -e "${YELLOW}Note: The free version of StarCraft II is sufficient for SMAC.${NC}"
            echo ""
            read -p "Press Enter to exit..."
            exit 0
            ;;
        3)
            echo -e "${YELLOW}Will install maps to $SC2PATH${NC}"
            mkdir -p "$SC2PATH"
            ;;
        *)
            echo -e "${RED}Exiting.${NC}"
            exit 0
            ;;
    esac
fi

echo -e "${GREEN}StarCraft II installation found at:${NC} $SC2PATH"

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
curl -L -o SMAC_Maps.zip https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip

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

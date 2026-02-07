#!/bin/bash

# VAVE AI - JSW MGI Restore Script
# This script restores from a backup

set -e

# Configuration
APP_DIR="/opt/vave-ai"
APP_USER="vave"
BACKUP_DIR="$APP_DIR/backups"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VAVE AI Restore Script${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# List available backups
echo -e "${YELLOW}Available backups:${NC}"
ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null | nl || {
    echo -e "${RED}No backups found in $BACKUP_DIR${NC}"
    exit 1
}

# Get backup file
echo -e "\n${YELLOW}Enter backup filename (or number from list):${NC}"
read -r BACKUP_INPUT

# Handle numeric input
if [[ "$BACKUP_INPUT" =~ ^[0-9]+$ ]]; then
    BACKUP_FILE=$(ls -1 "$BACKUP_DIR"/*.tar.gz | sed -n "${BACKUP_INPUT}p")
else
    BACKUP_FILE="$BACKUP_DIR/$BACKUP_INPUT"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Selected backup: $BACKUP_FILE${NC}"
echo -e "${RED}WARNING: This will restore from backup and may overwrite existing data!${NC}"
echo -e "${YELLOW}Continue? (yes/no):${NC}"
read -r CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Restore cancelled${NC}"
    exit 0
fi

# Stop service
echo -e "${YELLOW}Stopping application service...${NC}"
systemctl stop vave-ai.service || true

# Extract backup
TEMP_DIR=$(mktemp -d)
echo -e "${YELLOW}Extracting backup...${NC}"
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)
BACKUP_PATH="$TEMP_DIR/$BACKUP_NAME"

# Restore PostgreSQL database
if [ -f "$BACKUP_PATH/database.dump" ]; then
    echo -e "${YELLOW}Restoring PostgreSQL database...${NC}"
    source "$APP_DIR/.env"
    PGPASSWORD="${DB_PASSWORD:-${DB_PASS}}" pg_restore -h "${DB_HOST:-localhost}" \
        -U "${DB_USER:-vave_user}" \
        -d "${DB_NAME:-vave_db}" \
        --clean \
        --if-exists \
        "$BACKUP_PATH/database.dump"
fi

# Restore SQLite users database
if [ -f "$BACKUP_PATH/users.db" ]; then
    echo -e "${YELLOW}Restoring SQLite users database...${NC}"
    cp "$BACKUP_PATH/users.db" "$APP_DIR/users.db"
    chown "$APP_USER:$APP_USER" "$APP_DIR/users.db"
fi

# Restore directories
for archive in "$BACKUP_PATH"/*.tar.gz; do
    if [ -f "$archive" ]; then
        DIR_NAME=$(basename "$archive" .tar.gz)
        echo -e "${YELLOW}Restoring $DIR_NAME...${NC}"
        tar -xzf "$archive" -C "$APP_DIR"
    fi
done

# Restore configuration
if [ -f "$BACKUP_PATH/.env" ]; then
    echo -e "${YELLOW}Backup contains .env file. Restore it? (yes/no):${NC}"
    read -r RESTORE_ENV
    if [ "$RESTORE_ENV" = "yes" ]; then
        cp "$BACKUP_PATH/.env" "$APP_DIR/.env"
        chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
    fi
fi

# Set ownership
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

# Start service
echo -e "${YELLOW}Starting application service...${NC}"
systemctl start vave-ai.service

echo -e "\n${GREEN}Restore completed!${NC}"
echo -e "${YELLOW}Please verify the application is working correctly${NC}"


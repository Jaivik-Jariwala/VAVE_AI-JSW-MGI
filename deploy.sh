#!/bin/bash

# VAVE AI - JSW MGI Deployment Script for Ubuntu Server
# This script automates the deployment process on Ubuntu Server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="vave-ai"
APP_DIR="/opt/vave-ai"
APP_USER="vave"
SERVICE_NAME="vave-ai.service"
NGINX_CONF="/etc/nginx/sites-available/vave-ai"
NGINX_ENABLED="/etc/nginx/sites-enabled/vave-ai"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VAVE AI - JSW MGI Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Step 1: Update system packages
echo -e "${YELLOW}[1/10] Updating system packages...${NC}"
apt-get update
apt-get upgrade -y

# Step 2: Install required system packages
echo -e "${YELLOW}[2/10] Installing system dependencies...${NC}"
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    postgresql \
    postgresql-contrib \
    nginx \
    supervisor \
    git \
    curl \
    wget \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    ffmpeg \
    libsm6 \
    libxext6

# Step 3: Create application user
echo -e "${YELLOW}[3/10] Creating application user...${NC}"
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d "$APP_DIR" -m "$APP_USER"
    echo -e "${GREEN}User $APP_USER created${NC}"
else
    echo -e "${YELLOW}User $APP_USER already exists${NC}"
fi

# Step 4: Create application directory
echo -e "${YELLOW}[4/10] Setting up application directory...${NC}"
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/backups"

# Step 5: Copy application files (assuming script is run from project directory)
echo -e "${YELLOW}[5/10] Copying application files...${NC}"
CURRENT_DIR=$(pwd)
cp -r "$CURRENT_DIR"/* "$APP_DIR/" 2>/dev/null || true
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# Step 6: Setup Python virtual environment
echo -e "${YELLOW}[6/10] Setting up Python virtual environment...${NC}"
cd "$APP_DIR"
sudo -u "$APP_USER" python3.11 -m venv venv
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install --upgrade pip setuptools wheel
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install -r requirements.txt

# Step 7: Download NLTK data
echo -e "${YELLOW}[7/10] Downloading NLTK data...${NC}"
sudo -u "$APP_USER" "$APP_DIR/venv/bin/python" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Step 8: Setup PostgreSQL database
echo -e "${YELLOW}[8/10] Setting up PostgreSQL database...${NC}"
if [ ! -f "$APP_DIR/.env" ]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo -e "${RED}IMPORTANT: Please edit $APP_DIR/.env and set your database credentials and secret key!${NC}"
    echo -e "${YELLOW}Press Enter after editing .env file...${NC}"
    read
fi

# Load environment variables
source "$APP_DIR/.env"

# Create PostgreSQL database and user
sudo -u postgres psql <<EOF
-- Create database
CREATE DATABASE ${DB_NAME:-vave_db};

-- Create user
CREATE USER ${DB_USER:-vave_user} WITH PASSWORD '${DB_PASSWORD:-changeme}';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME:-vave_db} TO ${DB_USER:-vave_user};

-- Connect to database and grant schema privileges
\c ${DB_NAME:-vave_db}
GRANT ALL ON SCHEMA public TO ${DB_USER:-vave_user};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${DB_USER:-vave_user};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${DB_USER:-vave_user};
EOF

echo -e "${GREEN}PostgreSQL database configured${NC}"

# Step 9: Initialize database schema
echo -e "${YELLOW}[9/10] Initializing database schema...${NC}"
cd "$APP_DIR"
sudo -u "$APP_USER" "$APP_DIR/venv/bin/python" create_database.py || echo -e "${YELLOW}Database initialization skipped (may already exist)${NC}"

# Step 10: Create systemd service
echo -e "${YELLOW}[10/10] Creating systemd service...${NC}"
cat > "/etc/systemd/system/$SERVICE_NAME" <<EOF
[Unit]
Description=VAVE AI - JSW MGI Application
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=notify
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/gunicorn \\
    --workers 4 \\
    --bind 127.0.0.1:5000 \\
    --timeout 120 \\
    --access-logfile $APP_DIR/logs/access.log \\
    --error-logfile $APP_DIR/logs/error.log \\
    --log-level info \\
    app:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Step 11: Setup Nginx reverse proxy
echo -e "${YELLOW}[11/11] Configuring Nginx...${NC}"
cat > "$NGINX_CONF" <<EOF
server {
    listen 80;
    server_name _;  # Change this to your domain name

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /static {
        alias $APP_DIR/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /temp {
        alias $APP_DIR/temp;
        expires 1d;
    }
}
EOF

# Enable Nginx site
ln -sf "$NGINX_CONF" "$NGINX_ENABLED"
nginx -t && systemctl reload nginx

# Step 12: Set proper permissions
echo -e "${YELLOW}Setting file permissions...${NC}"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod +x "$APP_DIR/deploy.sh"
chmod +x "$APP_DIR/backup.sh"
chmod +x "$APP_DIR/health_check.sh"

# Final steps
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Edit $APP_DIR/.env and configure all settings"
echo -e "2. Ensure model files are in $APP_DIR/model/"
echo -e "3. Ensure images are in $APP_DIR/images/"
echo -e "4. Start the service: ${GREEN}systemctl start $SERVICE_NAME${NC}"
echo -e "5. Check status: ${GREEN}systemctl status $SERVICE_NAME${NC}"
echo -e "6. View logs: ${GREEN}journalctl -u $SERVICE_NAME -f${NC}"
echo -e "\n${YELLOW}Useful commands:${NC}"
echo -e "  Start:   systemctl start $SERVICE_NAME"
echo -e "  Stop:    systemctl stop $SERVICE_NAME"
echo -e "  Restart: systemctl restart $SERVICE_NAME"
echo -e "  Status:  systemctl status $SERVICE_NAME"
echo -e "  Logs:    journalctl -u $SERVICE_NAME -f"
echo -e "\n"


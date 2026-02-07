#!/bin/bash

# VAVE AI - JSW MGI Quick Start Script
# Simple script to start/stop/restart the service

SERVICE_NAME="vave-ai.service"

case "$1" in
    start)
        echo "Starting VAVE AI service..."
        sudo systemctl start "$SERVICE_NAME"
        sudo systemctl status "$SERVICE_NAME"
        ;;
    stop)
        echo "Stopping VAVE AI service..."
        sudo systemctl stop "$SERVICE_NAME"
        ;;
    restart)
        echo "Restarting VAVE AI service..."
        sudo systemctl restart "$SERVICE_NAME"
        sudo systemctl status "$SERVICE_NAME"
        ;;
    status)
        sudo systemctl status "$SERVICE_NAME"
        ;;
    logs)
        sudo journalctl -u "$SERVICE_NAME" -f
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac

exit 0


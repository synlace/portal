#!/bin/sh
set -e

# Create a mock sysctl command to bypass wg-quick permission errors inside Docker
if [ ! -f "/usr/local/bin/sysctl" ]; then
    echo "Creating mock sysctl command..."
    mkdir -p /usr/local/bin
    cat <<'EOF' > /usr/local/bin/sysctl
#!/bin/sh
# Mock sysctl to bypass permission denied errors in restricted environments
exit 0
EOF
    chmod +x /usr/local/bin/sysctl
fi

# Check if WireGuard config is provided as an environment variable
if [ -n "$WG_CONFIG" ]; then
    echo "Creating WireGuard configuration from WG_CONFIG environment variable..."
    mkdir -p /etc/wireguard
    echo "$WG_CONFIG" > /etc/wireguard/wg0.conf
    chmod 600 /etc/wireguard/wg0.conf
fi

# If wg0.conf exists, try to bring up the interface
if [ -f "/etc/wireguard/wg0.conf" ]; then
    echo "Starting WireGuard interface (wg0)..."
    # wg-quick needs NET_ADMIN capability to run successfully
    if wg-quick up wg0; then
        echo "WireGuard tunnel is UP!"
        # Show public IP to verify routing
        echo "Verifying public IP..."
        if curl --connect-timeout 5 -s ipinfo.io; then
            echo ""
        else
            echo "Warning: Could not fetch public IP to verify routing"
        fi
    else
        echo "Error: Failed to bring up WireGuard interface. Ensure the container has NET_ADMIN capability."
    fi
fi

# Execute the main CMD (FastAPI app)
exec "$@"

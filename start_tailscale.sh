#!/bin/bash
# Start tailscaled in userspace networking mode
nohup /usr/sbin/tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /var/log/tailscaled.log 2>&1 &
echo "Tailscale started in background."

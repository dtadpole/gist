#!/usr/bin/env bash
# (Re)establish ssh-localhost internet egress on this locked-down devserver.
# The fb-credentials cert clears and Chef overwrites the sshd bypass (~hourly),
# so this is idempotent and safe to re-run before any network step.
set -uo pipefail

fbwallet_fetch >/dev/null 2>&1 || true   # repopulate /var/facebook/credentials/$USER/ssh

if ! grep -q "Address 2401:db00::/32" /etc/ssh/sshd_config; then
  sudo cp /etc/ssh/sshd_config "/etc/ssh/sshd_config.bak.$(date +%Y%m%d-%H%M%S)"
  sudo sed -i '/^Match Group users LocalPort 22$/i\
# Devserver-to-devserver: publickey-only for zhenc from datacenter IPv6\
Match User zhenc LocalPort 22 Address 2401:db00::/32,127.0.0.1,::1\
  AuthenticationMethods publickey\
  AllowTcpForwarding yes\
' /etc/ssh/sshd_config
  sudo sshd -t && sudo systemctl reload sshd && echo "[ssh_fix] sshd bypass reapplied"
else
  echo "[ssh_fix] sshd bypass already present"
fi

if ssh -o BatchMode=yes -o ConnectTimeout=5 localhost "true" 2>/dev/null; then
  echo "[ssh_fix] ssh localhost OK"
else
  echo "[ssh_fix] ssh localhost STILL FAILING" >&2
  exit 1
fi

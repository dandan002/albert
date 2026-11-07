# AWS host (us-east-2, same region as Kalshi's exchange)

- Instance `albert` (m7i-flex.large, Ubuntu 24.04, encrypted 100 GB gp3, IMDSv2 only), Elastic IP
  16.59.235.153, security group `albert-ssh`: SSH (22) only from the owner's IP. Key pair `albert`
  = the owner's `~/.ssh/id_ed25519`. Clock: chrony on Amazon Time Sync (169.254.169.123).
- `ssh ubuntu@16.59.235.153`; code in `~/albert` (venv `.venv`), secrets `~/albert/.env` (mode 600).
- Services (systemd, `sudo systemctl status albert-*`):
  - `albert-record` (deploy/record-loop.sh --paired-days 3) → data/raw/stream
  - `albert-paper` (deploy/paper-loop.sh) → data/paper_l2
  - `albert-forward` (deploy/forward-loop.sh, LEADS="mentions fcs rain") → data/cache/forward
  - `albert-mentions@demo` (albert live mentions --env demo; args in deploy/mentions-demo.conf)
  - prod pilot later: `albert-mentions@prod` with deploy/mentions-prod.conf containing
    `MENTIONS_ARGS=--live ...` AND `KALSHI_LIVE_ENABLED=1` in .env (both required).
- Update code: from the Mac, `rsync -az --exclude data/ --exclude .venv/ --exclude .env ./ "ubuntu@16.59.235.153:albert/"`
  then `sudo systemctl restart albert-...`.
- Pull data back: `rsync -az "ubuntu@16.59.235.153:albert/data/raw/stream/" data/raw/stream/` (append-only).
- Security group IP changes (new home IP): `aws ec2 authorize-security-group-ingress --region us-east-2
  --group-id sg-04dc0331e14485794 --protocol tcp --port 22 --cidr <ip>/32` (and revoke the old one).

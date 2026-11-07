# AWS host (us-east-2, same region as Kalshi's exchange)

- Instance `albert` = `i-01990679def3bfdf4` (m7i-flex.large: the account is on the AWS Free plan,
  which only allows free-tier types; Ubuntu 24.04, encrypted 100 GB gp3, IMDSv2 only), Elastic IP
  16.59.235.153. Key pair `albert` = the owner's `~/.ssh/id_ed25519`. Clock: chrony on Amazon Time
  Sync (169.254.169.123).
- **Access: Systems Manager Session Manager only.** Security group `albert-ssh` has NO inbound rules
  (port 22 closed). Instance profile `albert-ec2-ssm` (AmazonSSMManagedInstanceCore). On the Mac:
  Session Manager plugin in `~/.local/sessionmanagerplugin` (symlink `~/.local/bin`), and
  `~/.ssh/config` host `albert` (ProxyCommand `aws ssm start-session ... AWS-StartSSHSession`,
  HostKeyAlias 16.59.235.153). So: `ssh albert`, `rsync ... albert:albert/`, or a plain shell with
  `aws ssm start-session --region us-east-2 --target i-01990679def3bfdf4`.
- Code in `~/albert` (venv `.venv`), secrets `~/albert/.env` (mode 600).
- Services (systemd, `sudo systemctl status albert-*`):
  - `albert-record` (deploy/record-loop.sh --paired-days 3) → data/raw/stream
  - `albert-paper` (deploy/paper-loop.sh) → data/paper_l2
  - `albert-forward` (deploy/forward-loop.sh, LEADS="mentions fcs rain") → data/cache/forward
  - `albert-mentions@demo` (albert live mentions --env demo; args in deploy/mentions-demo.conf)
  - prod pilot later: `albert-mentions@prod` with deploy/mentions-prod.conf containing
    `MENTIONS_ARGS=--live ...` AND `KALSHI_LIVE_ENABLED=1` in .env (both required).
- Update code: from the Mac, `rsync -az --exclude data/ --exclude .venv/ --exclude .env ./ albert:albert/`
  then `sudo systemctl restart albert-...`.
- Pull data back: `rsync -az albert:albert/data/raw/stream/ data/raw/stream/` (append-only).
- Emergency direct SSH (only if SSM is down): `aws ec2 authorize-security-group-ingress --region us-east-2
  --group-id sg-04dc0331e14485794 --protocol tcp --port 22 --cidr <your ip>/32`, then revoke it after.

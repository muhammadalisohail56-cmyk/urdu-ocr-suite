# Deploying the Urdu OCR Suite on a Google Compute Engine VM

Target: one small GCE VM running the FastAPI app (`main:app`) under **systemd**,
with **Caddy** providing automatic HTTPS via a free **DuckDNS** hostname.
Mode: **Gemini-only** (lean — no torch/UTRNet).

---

## ⚠️ Do this FIRST (security)

The Gemini key and a GitHub token were exposed in plaintext previously. Before
going live:

1. **Rotate the Gemini API key** in Google AI Studio (revoke the old one).
2. **Revoke the leaked GitHub token.**
3. Generate a **fresh `SESSION_SECRET`** (see step 6). Never reuse the dev one.

Deploy only with clean secrets.

---

## 1. Create the GCP project & VM (browser)

1. console.cloud.google.com → create/select a project → enable **billing**.
2. Enable the **Compute Engine API**.
3. Compute Engine → VM instances → **Create instance**:
   - **Machine type:** `e2-small` (2 GB) — enough for 2 users. Use `e2-medium`
     if PDF rendering feels tight.
   - **Boot disk:** Ubuntu 24.04 LTS, 20 GB.
   - **Firewall:** check **Allow HTTP** and **Allow HTTPS**.
4. Create it; note the **External IP** (you'll need it for DuckDNS).

## 2. Point DuckDNS at the VM

1. Go to duckdns.org, sign in, create a subdomain, e.g. `urdu-ocr`.
2. Set its IP to the VM's **External IP**. (Result: `urdu-ocr.duckdns.org`.)
3. Wait ~1 min, then from your laptop: `ping urdu-ocr.duckdns.org` should show the VM IP.

## 3. SSH into the VM

Use the **SSH** button in the GCE console (simplest), or `gcloud compute ssh <vm-name>`.

## 4. Install system packages

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git debian-keyring debian-archive-keyring apt-transport-https curl
# Install Caddy (official repo):
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

## 5. Create an app user and clone the repo

```bash
sudo adduser --disabled-password --gecos "" ocr
sudo su - ocr
git clone https://github.com/muhammadalisohail56-cmyk/urdu-ocr-suite.git
cd urdu-ocr-suite
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt        # lean base; NOT requirements-hybrid.txt
```

## 6. Create `.env` (rotated key + fresh secret)

Generate a strong session secret:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Generate each login credential. The app accepts `username:password` or, better,
a hashed form `username:sha256:<hex-digest>` — use the hashed form:
```bash
python3 -c "import hashlib,getpass; u=input('username: '); p=getpass.getpass('password: '); print(u+':sha256:'+hashlib.sha256(p.encode()).hexdigest())"
```

Then create the file:
```bash
cp .env.example .env
nano .env
```
Set (these are the real variable names from `.env.example`):
```
GEMINI_API_KEY=<your ROTATED key>
GEMINI_MODEL=gemini-2.5-flash
GEMINI_ONLY=1
USER1_CREDENTIALS=<username:sha256:hexdigest>
USER2_CREDENTIALS=<username:sha256:hexdigest>
SESSION_SECRET=<the token_urlsafe value>
SESSION_TTL_SECONDS=43200
COOKIE_SECURE=1          # we serve over HTTPS via Caddy
PDF_RENDER_DPI=200
# Cost estimate: flash is far cheaper than the pro defaults in .env.example.
RATE_INPUT_PER_M=0.30
RATE_OUTPUT_PER_M=2.50
DATABASE_URL=sqlite:///./ocr_suite.db
UPLOAD_DIR=storage/uploads
IMAGE_DIR=storage/pages
```
Lock it down:
```bash
chmod 600 .env
```

## 7. Smoke-test

```bash
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```
It should start cleanly. Ctrl-C to stop, then `exit` back to your sudo user.

## 8. Install the systemd service

```bash
sudo cp /home/ocr/urdu-ocr-suite/deploy/urdu-ocr.service /etc/systemd/system/urdu-ocr.service
sudo systemctl daemon-reload
sudo systemctl enable --now urdu-ocr
sudo systemctl status urdu-ocr        # should be "active (running)"
journalctl -u urdu-ocr -f             # live logs; Ctrl-C to exit
```

## 9. Configure Caddy (HTTPS)

Edit the Caddyfile to use your DuckDNS hostname:
```bash
sudo cp /home/ocr/urdu-ocr-suite/deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile         # replace your-subdomain.duckdns.org
sudo systemctl restart caddy
sudo systemctl status caddy
```
Caddy will automatically fetch a Let's Encrypt certificate (needs ports 80/443
open — they are, from step 1).

## 10. Verify

Open `https://urdu-ocr.duckdns.org` in a browser → you should get the login page
with a valid padlock. Log in with a credential from step 6 and upload a test PDF.

---

## Operations cheat-sheet

| Task | Command |
|---|---|
| Restart app | `sudo systemctl restart urdu-ocr` |
| App logs | `journalctl -u urdu-ocr -f` |
| Deploy new code | `sudo su - ocr`, `cd urdu-ocr-suite && git pull && ./venv/bin/pip install -r requirements.txt`, `exit`, `sudo systemctl restart urdu-ocr` |
| Caddy logs | `journalctl -u caddy -f` |
| Reload Caddy after edit | `sudo systemctl restart caddy` |

## Notes / gotchas

- **SQLite + local files:** the DB (`ocr_suite.db`) and uploads live on the VM
  disk. Back up the disk (or snapshot it) if the data matters. Do **not** move to
  a stateless/autoscaling host without migrating DB→managed SQL and files→object
  storage.
- **Cost control:** keep `GEMINI_MODEL=gemini-2.5-flash` (cheap). Pro's hidden
  "thinking" tokens caused a prior bill overrun.
- **Firewall:** only 80/443 (Caddy) need to be public. The app itself binds to
  `127.0.0.1:8000` and is not directly reachable.

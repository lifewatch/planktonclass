# Remote GPU Runbook

This file explains how to sync the PI10 code to `MOC-GPU-3` and start the remote GPU workers for `predict_gpu_remote.py`.

## Normal Workflow

Run these commands on your local Windows machine in PowerShell:

```powershell
cd D:\USERS\wout.decrop\environments\PI10
.\planktonclass\PI10\sync_and_start_remote.ps1
```

This syncs the local `planktonclass` folder to the GPU server and starts one worker per GPU in detached `screen` sessions.

The helper prefers `rsync` when it is installed. If `rsync` is missing on Windows, it automatically falls back to `tar.exe` + `scp` while still excluding `PI10/predict_gpu_config.json` and `.env` files.

Before starting workers, the helper checks whether the remote server can read `/mnt/qarchive_data_sensors/plankton-imager-10`. If not, it opens an interactive SSH repair step that runs `cifscreds add` and `sudo mount`; this may ask for your qarchive/sudo password. Use `-SkipQarchiveRepair` only if you want to skip that check.

## Private User Profile

The local helper reads remote login defaults from the private `PI10/predict_gpu_config.json`. This file is ignored by Git, so it can remember usernames and email addresses without putting them on GitHub. It must not contain passwords.

Example:

```json
"remote": {
  "active_profile": "wout",
  "profiles": {
    "wout": {
      "display_name": "Wout Decrop",
      "email": "wout.decrop@vliz.be",
      "ssh_user": "wout.decrop",
      "ssh_host": "MOC-GPU-3.vliz.be",
      "remote_project": "/data/woutdecrop/projects/planktonclass",
      "remote_env": "/data/woutdecrop/envs/planktonclass-gpu",
      "qarchive_user": "wout.decrop",
      "qarchive_domain": "vliz.be",
      "qarchive_root": "/mnt/qarchive_data_sensors",
      "qarchive_check_dir": "/mnt/qarchive_data_sensors/plankton-imager-10",
      "gpus": [0, 1]
    }
  }
}
```

For a new user, add another profile and change `active_profile` to that profile name. Passwords are still entered manually when SSH, `cifscreds`, or `sudo` prompts for them.

## One-Time SSH Setup

If the normal workflow asks for the SSH password every time, run this once:

```powershell
cd D:\USERS\wout.decrop\environments\PI10
.\planktonclass\PI10\sync_and_start_remote.ps1 -SetupSshKey
```

This installs your local SSH public key on the remote server. It may ask for the password once.

## Useful Local Commands

Sync only, without starting workers:

```powershell
.\planktonclass\PI10\sync_and_start_remote.ps1 -SyncOnly
```

Start remote workers only, without syncing:

```powershell
.\planktonclass\PI10\sync_and_start_remote.ps1 -StartOnly
```

Start only GPU 0:

```powershell
.\planktonclass\PI10\sync_and_start_remote.ps1 -Gpus 0
```

Start both GPUs:

```powershell
.\planktonclass\PI10\sync_and_start_remote.ps1 -Gpus 0,1
```

Skip the qarchive credential/mount repair step:

```powershell
.\planktonclass\PI10\sync_and_start_remote.ps1 -SkipQarchiveRepair
```

## What Starts On The Server

The remote launcher starts:

- `predict_gpu_gpu0`: runs `predict_gpu_remote.py` on GPU 0.
- `predict_gpu_gpu1`: runs `predict_gpu_remote.py` on GPU 1.

Each worker has its own temporary scratch folder:

```text
.../PI10_tempUntarred/worker_gpu0
.../PI10_tempUntarred/worker_gpu1
```

The workers share the same source/output folder, but they claim TAR files using atomic `.lock` files so they do not process the same TAR at the same time.

## Check Remote Status

Connect to the server:

```bash
ssh wout.decrop@MOC-GPU-3.vliz.be
```

List running screens:

```bash
screen -ls
```

Check GPU usage:

```bash
nvidia-smi
```

Check worker logs:

```bash
tail -n 80 /data/woutdecrop/projects/planktonclass/PI10/run_logs/predict_gpu_gpu0.log
tail -n 80 /data/woutdecrop/projects/planktonclass/PI10/run_logs/predict_gpu_gpu1.log
```

Reconnect to a worker:

```bash
screen -r predict_gpu_gpu0
screen -r predict_gpu_gpu1
```

The same output is written to the log file and printed live inside the `screen` session.

Detach again without stopping it:

```text
Ctrl + A, then D
```

## Manual Remote Start

If you are already SSH'd into the server, run:

```bash
cd /data/woutdecrop/projects/planktonclass/PI10
bash remote_start_gpu_workers.sh
```

## qarchive Mount

The remote launcher checks `/mnt/qarchive_data_sensors` and tries a non-interactive mount. If the server still needs credentials after a reboot, run this once on the server:

```bash
cifscreds add -u wout.decrop -d vliz.be
sudo mount /mnt/qarchive_data_sensors/
```

If the workers appear briefly and then disappear from `screen -ls`, check the logs:

```bash
tail -n 80 /data/woutdecrop/projects/planktonclass/PI10/run_logs/predict_gpu_gpu0.log
```

If the log says `PermissionError: [Errno 13] Permission denied: '/mnt/qarchive_data_sensors/...'`, refresh the qarchive credentials with the two commands above, then rerun the local sync/start helper.

## Files In This Workflow

- `sync_and_start_remote.ps1`: run this locally from PowerShell to sync and start.
- `remote_start_gpu_workers.sh`: run this on the remote server to start screens.
- `predict_gpu_remote.py`: the actual processing script.
- `syncornize.txt`: short sync notes.
- `start_server.txt`: short server-start notes.

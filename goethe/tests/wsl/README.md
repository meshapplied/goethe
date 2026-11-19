# Nested Windows VM harness for install.ps1

Iterate the Windows-to-WSL bootstrap without melting the host laptop. CUDA and the Windows NVIDIA driver are not validated here; use a real Windows+GPU box only for that acceptance.

## Caps (host-safe)

| Resource | Value |
|---|---|
| vCPUs | 4 |
| RAM | 8 GB |
| Disk | 64 GB sparse qcow2 |
| Snapshot | `pre-wsl` before every installer run |

Leave at least 2 to 4 host cores and several GB RAM free.

## One-time host prep (Fedora / Ubuntu)

```bash
# Nested virt (Intel shown; AMD: kvm_amd nested=1)
echo Y | sudo tee /sys/module/kvm_intel/parameters/nested
cat /sys/module/kvm_intel/parameters/nested   # expect Y

sudo dnf install -y qemu-kvm libvirt virt-manager edk2-ovmf swtpm   # Fedora
# sudo apt install -y qemu-system libvirt-daemon-system virt-manager ovmf swtpm qemu-utils

sudo usermod -aG libvirt "$USER"
sudo virt-host-validate qemu
```

## Create the guest

Prefer quickemu (https://github.com/quickemu-project/quickemu) or virt-manager (Q35, UEFI, TPM, nested vmx/svm, Hyper-V enlightenments). Example with quickemu:

```bash
quickget windows 11
# Edit the generated conf: ram="8G" cpu_cores="4"
# Nested WSL2 often needs CPU flags such as:
#   extra_args="-cpu host,+vmx,hv_passthrough"   # Intel; tune for your host
quickemu --vm windows-11.conf
```

Inside Windows: finish OOBE, install VirtIO drivers if needed, then create a snapshot from the Linux host before testing:

```bash
quickemu --vm windows-11.conf --snapshot create pre-wsl
```

## Exercise install.ps1

1. Revert to `pre-wsl` if the guest is dirty.
2. Copy or serve `cdn/install.ps1` into the guest (HTTP from the host, shared folder, or paste).
3. Elevated PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
# Against a local CDN tree:
$env:GOETHE_PKG_URL = 'http://<host-lan-ip>:8080'
irm http://<host-lan-ip>:8080/install.ps1 | iex
# Or: powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

4. Expect: feature enable, possible reboot and RunOnce resume, Ubuntu-24.04, systemd, Linux install.sh, terminal with goethe.
5. From Windows: after `goethe up` inside WSL, hit `http://127.0.0.1:<port>`.

## What this proves / does not

| Check | Nested CPU VM | Bare-metal + NVIDIA |
|---|---|---|
| Feature enable, reboot/resume, Ubuntu, systemd | Yes | Yes |
| Linux installer + `goethe doctor` (CPU) | Yes | Yes |
| Terminal open / localhost relay | Yes | Yes |
| Driver >=580, `/usr/lib/wsl/lib`, pack load | No | Yes |

## Helper

`./prepare_host.sh` only checks nested virt and prints the caps; it never downloads a Windows ISO or starts a VM.

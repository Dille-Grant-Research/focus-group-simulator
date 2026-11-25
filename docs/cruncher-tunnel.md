# Cruncher tunnel setup (macOS / Linux)

This forwards `localhost:1234` on your machine to LM Studio on `cruncher.mnstate.edu:1234`. The connection goes through `broker.mnstate.edu`, then `smoke.mnstate.edu`.

> **Note:** in practice the tunnel stays up for about **1 hour**, then drops. `ControlPersist 8h` in the config does not keep it alive longer. When it drops, run the open command again.

## 1. Back up your current SSH config
Skip this step if you don't have a config yet.
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/config ] && cp ~/.ssh/config ~/.ssh/config.bak
```

## 2. Add the two host entries
Replace `YOUR_MSUM_StarID` in both places.
```bash
cat >> ~/.ssh/config <<'EOF'

Host msum-broker
  HostName broker.mnstate.edu
  User YOUR_MSUM_StarID

Host msum-cruncher-tunnel
  HostName smoke.mnstate.edu
  User YOUR_MSUM_StarID
  ProxyJump msum-broker
  LocalForward 1234 cruncher.mnstate.edu:1234
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
  ServerAliveInterval 60
  ExitOnForwardFailure yes
EOF
chmod 600 ~/.ssh/config
```

## 3. Open the tunnel
You'll be asked for your password and MFA for each hop.
```bash
ssh -fN msum-cruncher-tunnel
```

## 4. Test it
This should return JSON listing the loaded model.
```bash
curl http://localhost:1234/v1/models
```

## 5. Check, close, or reopen the tunnel
```bash
ssh -O check msum-cruncher-tunnel   # is it still up?
ssh -O exit  msum-cruncher-tunnel   # close it
ssh -fN msum-cruncher-tunnel        # reopen after it drops (~1h)
```

## 6. Point the app at it
- Backend running locally: `LLM_BASE_URL=http://localhost:1234`
- Backend running in Docker: `LLM_BASE_URL=http://host.docker.internal:1234`

## If something goes wrong
- **"Address already in use":** something else is using port 1234. Find it with `lsof -i :1234`, or change both `LocalForward 1234` and `LLM_BASE_URL` to another port, such as 1235.
- **To see why it dropped:** open the tunnel with a log, `ssh -E ~/cruncher-tunnel.log -fN msum-cruncher-tunnel`, then check `~/cruncher-tunnel.log`.
- **To undo everything:** `cp ~/.ssh/config.bak ~/.ssh/config`

## Windows (built-in OpenSSH)
Put the config in `C:\Users\<you>\.ssh\config`. Remove the `ControlMaster`, `ControlPath` and `ControlPersist` lines, because Windows OpenSSH doesn't support them. Run `ssh -N msum-cruncher-tunnel` in a terminal and leave that window open while you work.

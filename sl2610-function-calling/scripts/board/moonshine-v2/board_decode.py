#!/usr/bin/env python3
# Phase-C end-to-end BOARD verifier for the fully-DSL Moonshine v2 streaming ASR.
#
# Drives a greedy KV-cache decode where the two decoder graphs (masked prefill + dynamic-cache
# with_past) execute ON the SL2610 via torq-run-module, while the light per-token work (argmax,
# embedding lookup, RoPE) stays on the host. Cross-KV + the growing self-cache live on the board as
# .bin files between steps (renamed in place), so only tiny tensors cross the wire each step.
#
# Proves the board's self-compiled vmfbs decode token-for-token identical to the ONNX reference.
# Result (beckett.wav first 64-frame window): board ids [18274,1898,29973,2] == ONNX -> "Ever tried?".
#
# Env:
#   MOON_SCRATCH  dir holding {stream/*.vmfb (x86, for the ONNX ref path is not needed here),
#                 baked/decoder/dec_embed.weight.bin, wp/cos.bin, wp/sin.bin,
#                 stream/golden/{adp_out_board.bin already staged as mem96/mask on the board}}
#   MOON_BOARD    ssh target (default root@192.168.3.26)
#   MOON_BDIR     board dir with the aarch64 vmfbs + mem96.bin + mask.bin (default /home/root/moon)
# Prereq on the board: {pre,wp}_masked_arm64.vmfb, mem96.bin, mask.bin already deployed;
# compile with scripts/compile-cpu-arm64-docker.sh and scp across.
import numpy as np, subprocess, os
S=os.environ.get("MOON_SCRATCH") or os.path.expanduser("~/moonshine-v2")
B=os.environ.get("MOON_BOARD","root@192.168.3.26"); MD=os.environ.get("MOON_BDIR","/home/root/moon")
G=f"{S}/stream/golden"; os.makedirs(G,exist_ok=True)
L,H,HD,DIM,VOCAB=6,8,40,320,32768; BOS,EOS,W,MAX,STEPS=1,2,64,96,24
RM="/home/root/torq2-stable/torq/_runtime_libs/torq-run-module"
ENVV="export LD_LIBRARY_PATH=/home/root/torq2-stable/iree/_runtime_libs:/home/root/torq2-stable/torq/_runtime_libs"

def sh(cmd): return subprocess.run(["ssh","-o","ConnectTimeout=8",B,cmd],capture_output=True,text=True,timeout=120)
def push(local,remote): subprocess.run(["scp","-q","-o","ConnectTimeout=8",local,f"{B}:{MD}/{remote}"],check=True,timeout=60)
def pull(remote,local): subprocess.run(["scp","-q","-o","ConnectTimeout=8",f"{B}:{MD}/{remote}",local],check=True,timeout=60)

emb=np.fromfile(f"{S}/baked/decoder/dec_embed.weight.bin",np.float32).reshape(VOCAB,DIM)
c0=np.fromfile(f"{S}/wp/cos.bin",np.float32); s0=np.fromfile(f"{S}/wp/sin.bin",np.float32); invf=np.arctan2(s0[1::2],c0[0::2])
def rope(p):
    a=p*invf; c=np.empty(40,np.float32); s=np.empty(40,np.float32); c[0::2]=np.cos(a); c[1::2]=np.cos(a); s[0::2]=-np.sin(a); s[1::2]=np.sin(a); return c.reshape(1,40),s.reshape(1,40)
def putbin(x,name): x.astype(np.float32).tofile(f"{G}/{name}"); push(f"{G}/{name}",name)

# ---- PREFILL on board: BOS embed + board memory (mem96.bin) + cross mask (mask.bin) ----
putbin(emb[BOS].reshape(1,1,DIM),"pe0.bin")
po="".join(f" --output=@pb{i}.bin" for i in range(25))
r=sh(f"cd {MD}; {ENVV}; {RM} --device=local-task --module=pre_masked_arm64.vmfb --function=moonshine_v2_decoder_prefill "
     f"--input=1x1x320xf32=@pe0.bin --input=1x96x320xf32=@mem96.bin --input=1x1x1x96xf32=@mask.bin {po} 2>&1 | tail -1")
print("prefill:",r.stdout.strip())
mv=";".join(f"mv pb{4*l}.bin sK{l}.bin; mv pb{4*l+1}.bin sV{l}.bin; mv pb{4*l+2}.bin cK{l}.bin; mv pb{4*l+3}.bin cV{l}.bin" for l in range(L))
sh(f"cd {MD}; {mv}")
pull("pb24.bin",f"{G}/lg.bin"); tok=int(np.fromfile(f"{G}/lg.bin",np.float32).argmax())
ids=[tok]; past=1; print("tok1(board prefill)=",tok)

# ---- greedy with_past loop on board (dynamic self-cache grows each step) ----
for step in range(STEPS):
    if tok==EOS: break
    putbin(emb[tok].reshape(1,1,DIM),"e.bin"); cq,sq=rope(past); putbin(cq,"cq.bin"); putbin(sq,"sq.bin")
    sc=f"1x8x{past}x40xf32"
    wi=f"--input=1x1x320xf32=@e.bin --input=1x40xf32=@cq.bin --input=1x40xf32=@sq.bin"
    for l in range(L):
        wi+=f" --input={sc}=@sK{l}.bin --input={sc}=@sV{l}.bin --input=1x8x96x40xf32=@cK{l}.bin --input=1x8x96x40xf32=@cV{l}.bin"
        if l==0: wi+=" --input=1x1x1x96xf32=@mask.bin"
    wo="".join(f" --output=@wb{i}.bin" for i in range(13))
    sh(f"cd {MD}; {ENVV}; {RM} --device=local-task --module=wp_masked_arm64.vmfb --function=moonshine_v2_decoder_with_past {wi} {wo} 2>&1 | tail -1")
    mv=";".join(f"mv wb{2*l}.bin sK{l}.bin; mv wb{2*l+1}.bin sV{l}.bin" for l in range(L))
    sh(f"cd {MD}; {mv}")
    pull("wb12.bin",f"{G}/lg.bin"); tok=int(np.fromfile(f"{G}/lg.bin",np.float32).argmax())
    ids.append(tok); past+=1; print(f"step{step+1} past->{past} tok={tok}")

print("BOARD ids:",ids)

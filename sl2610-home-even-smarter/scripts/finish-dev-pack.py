"""Turns staged recipe outputs into an installable, UNSIGNED development pack_dir: descriptor.json from the blueprint's
descriptor_template and target (as materializeCartridge resolves it, minus measurements), and manifest.json
(schema 2: every artifact with role, size and sha256), so the app's CartridgeStore lists it and its verify checks it.

  finish-dev-pack.py <pack-dir> <blueprint.json> <target-id> <cartridge-id> [--lang en ...] [--device graph=device ...]

--device entries go to descriptor.attributes.devices, which the board's LinuxEngineFactory reads to place each graph
or stage on an IREE device (local-task = CPU, torq = NPU, vulkan = GPU); absent ones run on local-task.
"""
import argparse, hashlib, json, os

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("blueprint"); ap.add_argument("target"); ap.add_argument("id")
ap.add_argument("--lang", action="append", default=[])
ap.add_argument("--device", action="append", default=[])
a = ap.parse_args()

bp = json.load(open(a.blueprint))
target = next(t for t in bp["targets"] if t["id"] == a.target)
descriptor = {"spec_version": bp.get("spec_version", "0.4"), "id": a.id, "version": "0.0.0-dev"}
descriptor.update(bp["descriptor_template"])
descriptor["license"] = bp.get("license", "")
descriptor["target"] = target["target"]
if "requirements" in target:
    descriptor["requirements"] = target["requirements"]
attrs = dict(descriptor.get("attributes") or {})
if a.lang:
    attrs["languages"] = a.lang
if a.device:
    attrs["devices"] = dict(d.split("=", 1) for d in a.device)
# a target that exports its graphs with another chunk (params.chunk) says so where the engine reads it
if "chunk" in target.get("params", {}) and "runtime_hint" in attrs:
    attrs["runtime_hint"] = dict(attrs["runtime_hint"], chunk=int(target["params"]["chunk"]))
descriptor["attributes"] = attrs
descriptor["note"] = f"unsigned development pack from the {a.target} recipe outputs (not materialized, not measured)"
with open(os.path.join(a.pack, "descriptor.json"), "w") as f:
    json.dump(descriptor, f, indent=2); f.write("\n")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

artifacts = []
for root, _, files in os.walk(a.pack):
    for name in sorted(files):
        full = os.path.join(root, name)
        rel = os.path.relpath(full, a.pack)
        if rel == "manifest.json":
            continue
        role = "descriptor" if rel == "descriptor.json" else rel.split(os.sep)[1]
        artifacts.append({"role": role, "path": rel, "digest": "sha256:" + sha256(full), "size": os.path.getsize(full)})
artifacts.sort(key=lambda x: x["path"])
manifest = {"schema_version": 2, "id": a.id, "version": "0.0.0-dev", "digest_alg": "sha256", "artifacts": artifacts}
with open(os.path.join(a.pack, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2); f.write("\n")
print(f"{a.id}: {len(artifacts)} artifacts, {sum(x['size'] for x in artifacts) / 1e6:.0f} MB, devices {attrs.get('devices', {})}")

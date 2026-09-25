#!/usr/bin/env python3
"""Phase-conditioned, periodic SiO2 score generation. Positions and sigma: Angstrom.

One shared scalar/vector equivariant GNN, trained on both glass and crystal.
Predicts -sigma * grad log p_sigma on an orthorhombic torus. This is a
structure generator, not an energy/force model or an equilibrium MD sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import ase.io
from ase import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import primitive_neighbor_list
import numpy as np
import torch
from torch import nn

FORMAT = "test40-phase-torus-v2"
PHASES = ("glass", "crystal")
SI_MASS, O_MASS = 28.0855, 15.9994
BEAD_MASS = SI_MASS + 2 * O_MASS
STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_pt(path, data):
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def load_pt(path):
    # Only load checkpoints produced locally or obtained from a trusted source.
    return torch.load(path, map_location="cpu", weights_only=False)


def output_dir(path, resume=False):
    if resume:
        if not path.is_dir():
            raise ValueError("Resume directory does not exist")
    elif path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty: {path}; use a new directory or --resume")
    path.mkdir(parents=True, exist_ok=True)
    return path


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def lengths_of(cell):
    cell = np.asarray(cell)
    lengths = np.diag(cell)
    if (not np.isfinite(cell).all() or np.any(lengths <= 0)
            or not np.allclose(cell, np.diag(lengths), atol=1e-6)):
        raise ValueError("Only positive, axis-aligned orthorhombic periodic cells are supported")
    return lengths


def tetrahedral_mapping(atoms, cutoff=2.2, allow_sharing_defects=False):
    """Shared-O mass-weighted tetrahedron centers; strict corner-sharing SiO2.

    Every Si has four O, every O belongs to two Si. Half of each oxygen mass
    belongs to each tetrahedron, so the overlapping units conserve total mass.
    """
    box = lengths_of(atoms.cell)
    if cutoff >= min(box) / 2:
        raise ValueError("Si-O mapping cutoff exceeds the minimum-image limit")
    z = atoms.numbers
    i, j, d = primitive_neighbor_list("ijD", pbc=atoms.pbc, cell=atoms.cell,
                                     positions=atoms.positions, cutoff=cutoff)
    mask = z[i] != z[j]
    degree = np.bincount(i[mask], minlength=len(atoms))
    valid_oxygen = np.all(degree[z == 8] >= 1) if allow_sharing_defects else np.all(degree[z == 8] == 2)
    if not np.all(degree[z == 14] == 4) or not valid_oxygen:
        bad_si = int(np.sum(degree[z == 14] != 4))
        bad_o = int(np.sum(degree[z == 8] != 2))
        raise ValueError(f"SiO4 mapping requires four O per Si and two Si per O; invalid Si={bad_si}, O={bad_o}. Check the reference/cutoff; defects are not silently remapped.")
    centers, sites = [], []
    for si in np.flatnonzero(z == 14):
        edges = np.flatnonzero(mask & (i == si))
        # Each displacement unwraps its O relative to the central Si.
        weights = O_MASS / degree[j[edges]]
        mass = SI_MASS + weights.sum()
        centers.append((atoms.positions[si] + (weights[:, None] * d[edges]).sum(0) / mass) % box)
        oxy = sorted(int(x) for x in j[edges])
        sites.append(dict(si_index=int(si), oxygen_indices=oxy,
                          oxygen_sharing_counts=[int(degree[x]) for x in oxy], mass_amu=float(mass)))
    return np.asarray(centers, dtype=np.float32), sites


def prepare(args):
    out = output_dir(args.output)
    metadata = dict(format=FORMAT, phases={}, units="Angstrom", representation=args.representation)
    for phase in PHASES:
        frames, cell_lengths, seen, sources = [], [], set(), []
        numbers, mapping = None, None
        # Each phase can come from a different pipeline (e.g. glass from a
        # single-frame lammps-data quench, crystal from a lammps-dump-text
        # MD trajectory), so format/type mapping is per-phase, not shared.
        input_format = getattr(args, phase + "_input_format")
        lammps_types = getattr(args, phase + "_lammps_types")
        index = getattr(args, phase + "_index")
        for path in getattr(args, phase):
            sources.append(dict(name=path.name, sha256=digest(path)))
            options = {}
            if lammps_types:
                if input_format == "lammps-data":
                    options["Z_of_type"] = dict(enumerate(lammps_types, 1))
                elif input_format == "lammps-dump-text":
                    # Dump files carry no mass/element info, only a bare
                    # type id, so ASE needs the type-order species list
                    # (specorder) instead of the Z_of_type mapping used for
                    # lammps-data.
                    options["specorder"] = [chemical_symbols[z] for z in lammps_types]
            # ase.io.iread's streaming reader refuses a non-trivial slice
            # start for chunk-scanned formats like lammps-dump-text (it
            # only allows start 0 or -1); ase.io.read has no such
            # restriction and the whole dataset is materialized in memory
            # right after this anyway, so there is no streaming benefit lost.
            result = ase.io.read(path, index=index, format=input_format, **options)
            for atoms in result if isinstance(result, list) else [result]:
                atoms = atoms.repeat(getattr(args, "repeat_" + phase))
                if not atoms.pbc.all():
                    raise ValueError(f"{path}: three periodic directions required")
                cell = lengths_of(atoms.cell)
                z = atoms.numbers
                if set(z) != {8, 14} or np.sum(z == 8) != 2 * np.sum(z == 14):
                    raise ValueError(f"{path}: expected Si:O = 1:2 (atomic numbers 14 and 8)")
                if numbers is None:
                    numbers = z.copy()
                if not np.array_equal(z, numbers):
                    raise ValueError(f"{phase}: all frames must have the same atom order")
                pos = np.asarray(atoms.get_positions() % cell, dtype=np.float32)
                if not np.isfinite(pos).all():
                    raise ValueError("Non-finite coordinates")
                if args.representation == "tetrahedron":
                    pos, sites = tetrahedral_mapping(atoms, args.mapping_cutoff, args.allow_sharing_defects)
                    if mapping is not None and sites != mapping and not args.allow_topology_drift:
                        raise ValueError(f"{phase}: Si-O topology changed between frames; this fixed-topology mapping cannot represent bond exchange (independently generated replicas, e.g. separately quenched glasses, have different networks; pass --allow-topology-drift if that is intentional)")
                    # Each frame's own bead positions (pos) are always computed
                    # fresh above from that frame's own topology, so training
                    # is unaffected by drift: bead i always is Si atom i's
                    # tetrahedron in whichever frame it is drawn from. Only
                    # the single stored mapping/bead_masses_amu (used for
                    # generate()'s output masses, not for training) reflects
                    # the LAST frame seen rather than being exactly right for
                    # every frame when --allow-topology-drift is set.
                    mapping = sites
                key = hashlib.sha256(pos.tobytes()).hexdigest()
                if key not in seen:
                    frames.append(pos)
                    cell_lengths.append(cell)
                    seen.add(key)
        if not frames:
            raise ValueError(f"No frames for {phase}")
        if len(frames) == 1:
            if not args.allow_single_reference:
                raise ValueError(f"{phase}: one unique frame; provide independent frames or explicitly use --allow-single-reference")
            train_ids = valid_ids = [0]
            mode = "single-reference reconstruction; validation is NOT independent"
        else:
            nvalid = max(1, math.ceil(len(frames) * args.validation_fraction))
            split = len(frames) - nvalid
            train_ids, valid_ids = list(range(split - args.split_gap)), list(range(split, len(frames)))
            if not train_ids:
                raise ValueError("Too few training frames for validation fraction / split gap")
            mode = "ordered held-out tail; correlation depends on supplied frame spacing"
        np.save(out / f"{phase}.npy", np.stack(frames))
        output_numbers = [14] * len(mapping) if mapping is not None else numbers.tolist()
        # One cell per frame (e.g. NPT "cell breathing"): the model already
        # takes box lengths as per-call conditioning, so training/generation
        # look these up per frame instead of assuming one fixed phase cell.
        metadata["phases"][phase] = dict(numbers=output_numbers,
            lengths=[c.tolist() for c in cell_lengths],
            train_ids=train_ids, valid_ids=valid_ids, validation_mode=mode,
            frames=len(frames), sha256=digest(out / f"{phase}.npy"), sources=sources,
            representation=args.representation, source_numbers=numbers.tolist(), mapping=mapping,
            ideal_bead_mass_amu=BEAD_MASS if mapping is not None else None,
            bead_masses_amu=[s["mass_amu"] for s in mapping] if mapping is not None else None,
            allow_sharing_defects=args.allow_sharing_defects,
            mapping_cutoff_A=args.mapping_cutoff if mapping is not None else None,
            mapping_definition="SiO4 center with each O mass divided by its sharing count; Si label denotes a CG bead" if mapping is not None else "all atom")
        if mapping is not None:
            abnormal = {o: n for s in mapping for o, n in zip(s["oxygen_indices"], s["oxygen_sharing_counts"]) if n != 2}
            metadata["phases"][phase]["nonbridging_or_overcoordinated_oxygen"] = abnormal
            print(f"  O with sharing count != 2: {len(abnormal)}; total CG mass: {sum(s['mass_amu'] for s in mapping):.4f} amu", flush=True)
        print(f"{phase}: {len(frames)} unique frames, {len(output_numbers)} sites ({args.representation}); {mode}", flush=True)
    save_json(out / "metadata.json", metadata)
    return 0


def load_dataset(path):
    meta = json.loads((path / "metadata.json").read_text())
    if meta["format"] != FORMAT:
        raise ValueError("Not a test40 dataset")
    arrays = {}
    for phase, info in meta["phases"].items():
        file = path / f"{phase}.npy"
        if digest(file) != info["sha256"]:
            raise ValueError(f"Dataset changed: {file}")
        arrays[phase] = np.load(file, mmap_mode="r")
    return arrays, meta


def graph(pos, lengths, cutoff):
    """Rebuild the actual noisy graph, identically in training and sampling."""
    box = np.asarray(lengths, dtype=float)
    if cutoff >= box.min() / 2:
        raise ValueError("cutoff must be strictly smaller than half the shortest cell side; replicate the reference")
    i, j, disp = primitive_neighbor_list("ijD", pbc=(True, True, True), cell=np.diag(box),
        positions=pos.detach().cpu().numpy(), cutoff=cutoff, self_interaction=False)
    return (torch.as_tensor(i, device=pos.device), torch.as_tensor(j, device=pos.device),
            torch.as_tensor(disp, dtype=pos.dtype, device=pos.device))


class VectorBlock(nn.Module):
    def __init__(self, width, radial):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3 * width))
        self.filter = nn.Linear(radial, 3 * width)
        self.mix_u = nn.Linear(width, width, bias=False)
        self.mix_v = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 3 * width))

    def forward(self, h, v, i, j, unit, radial, envelope):
        a, b, c = (self.message(h[j]) * self.filter(radial) * envelope[:, None]).chunk(3, -1)
        dh = torch.zeros_like(h).index_add(0, i, a)
        dv = torch.zeros_like(v).index_add(0, i, b[:, None, :] * unit[:, :, None] + c[:, None, :] * v[j])
        # Fixed normalization is shared by train/generate; no hidden cutoff changes.
        h, v = h + dh / math.sqrt(32), v + dv / math.sqrt(32)
        u, w = self.mix_u(v), self.mix_v(v)
        norm = torch.sqrt(w.square().sum(1) + 1e-8)
        a, b, c = self.update(torch.cat((h, norm), -1)).chunk(3, -1)
        return h + a + b * (u * w).sum(1), v + c[:, None, :] * u


class PhaseScore(nn.Module):
    """Scalar/vector message passing inspired by PaiNN; not a NequIP checkpoint."""
    def __init__(self, width=64, layers=3, cutoff=5.0, radial=16):
        super().__init__()
        self.cutoff = cutoff
        self.species = nn.Embedding(2, width)
        self.phase = nn.Embedding(2, width)
        self.condition = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width))
        self.register_buffer("centers", torch.linspace(0, cutoff, radial))
        self.blocks = nn.ModuleList([VectorBlock(width, radial) for _ in range(layers)])
        self.head = nn.Linear(width, 1, bias=False)

    def forward(self, types, edges, phase, sigma, lengths):
        i, j, disp = edges
        r = disp.norm(dim=-1)
        unit = disp / r.clamp_min(1e-8)[:, None]
        radial = torch.exp(-((r[:, None] - self.centers) / (self.cutoff / len(self.centers))) ** 2)
        envelope = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1)
        envelope = envelope * (r < self.cutoff)
        # Sorted box lengths and density are rotation-invariant global conditioning.
        box = sorted(float(x) for x in lengths)
        cond = disp.new_tensor([math.log(sigma), *[math.log(x) for x in box],
                                math.log(len(types) / math.prod(box))])
        # Re-added after every block. DM2's own production model re-injects
        # its condition embedding (cooling rate) after every conv layer this
        # way; DM2 does not feed sigma to the network at all (only geometry),
        # so this is not literally DM2's design ported over -- it applies
        # the same layer-wise re-injection *pattern* (also present in DM2's
        # code as NequIP_TimeEmbed, unused by DM2's own training script) to
        # sigma instead, since sigma is the more central conditioning
        # variable here (the target's character changes qualitatively with
        # it, unlike DM2's cooling rate). This is an untested hypothesis, not
        # a confirmed fix: with only the initial injection, sigma and cell
        # conditioning must survive 'layers' rounds of message passing
        # without the network being pushed to preserve it, but whether that
        # actually degrades it in just 3 layers is unverified. Safe for
        # equivariance regardless, since h (unlike v) is a pure
        # rotation-invariant scalar channel.
        cond_embed = self.condition(cond)
        h = self.species(types) + self.phase.weight[phase] + cond_embed
        v = h.new_zeros((len(types), 3, h.shape[-1]))
        for block in self.blocks:
            h, v = block(h, v, i, j, unit, radial, envelope)
            h = h + cond_embed
        return self.head(v).squeeze(-1)


def wrapped_target(noisy, clean, lengths, sigma):
    """Exact periodic Gaussian -sigma*score, with convergent image/Fourier sums."""
    box = torch.as_tensor(lengths, dtype=noisy.dtype, device=noisy.device)
    delta = (noisy - clean + box / 2) % box - box / 2
    result = torch.empty_like(delta)
    for axis in range(3):
        side = box[axis]
        d = delta[:, axis:axis + 1]
        if sigma / float(side) < 0.2:
            images = d + torch.arange(-2, 3, device=noisy.device) * side
            weights = torch.softmax(-0.5 * (images / sigma).square(), -1)
            result[:, axis] = (weights * images).sum(-1) / sigma
        else:
            k = torch.arange(1, 13, dtype=noisy.dtype, device=noisy.device)
            amplitude = torch.exp(-2 * math.pi**2 * k.square() * (sigma / side)**2)
            angle = 2 * math.pi * d * k / side
            density = 1 + 2 * (amplitude * torch.cos(angle)).sum(-1)
            deriv = -(4 * math.pi / side) * (k * amplitude * torch.sin(angle)).sum(-1)
            result[:, axis] = -sigma * deriv / density.clamp_min(1e-12)
    return result


def deadline(args):
    return time.monotonic() + args.time_budget_hours * 3600 if args.time_budget_hours else math.inf


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu for local checks")
    return torch.device(name)


def train(args):
    arrays, meta = load_dataset(args.dataset)
    device = device_for(args.device)
    config = dict(width=args.width, layers=args.layers, cutoff=args.cutoff)
    for info in meta["phases"].values():
        if args.cutoff >= min(min(cell) for cell in info["lengths"]) / 2:
            raise ValueError("Replicate input cells or reduce cutoff to below half the shortest side, in every frame")
    maximum = max(max(cell) for p in meta["phases"].values() for cell in p["lengths"])
    sigma_max = args.sigma_max or maximum
    if sigma_max <= args.sigma_min or math.exp(-2 * math.pi**2 * (sigma_max / maximum)**2) > 1e-5:
        raise ValueError("sigma-max must exceed sigma-min and be large enough for a uniform terminal distribution")
    settings = dict(dataset_sha256=digest(args.dataset / "metadata.json"), architecture=config,
        sigma_min=args.sigma_min, sigma_max=sigma_max, learning_rate=args.learning_rate,
        batch_size=args.batch_size, seed=args.seed, device=args.device)
    output = output_dir(args.output, args.resume)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PhaseScore(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    completed, history = 0, []
    if args.resume:
        ck = load_pt(output / "checkpoint.pt")
        if ck.get("format") != FORMAT or ck["settings"] != settings:
            raise ValueError("Resume requires the same dataset and training settings")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        completed, history = ck["step"], ck["history"]
    if args.updates < completed:
        raise ValueError("updates is below the completed checkpoint step")
    until = deadline(args)

    def save():
        save_pt(output / "checkpoint.pt", dict(format=FORMAT, settings=settings, metadata=meta,
            model=model.state_dict(), optimizer=optimizer.state_dict(), step=completed,
            history=history, rng=rng_state()))
        save_json(output / "training.json", dict(settings=settings, step=completed, history=history))

    def loss_for(phase, frame, sigma):
        info = meta["phases"][phase]
        lengths = info["lengths"][frame]
        clean = torch.tensor(np.array(arrays[phase][frame]), device=device)
        box = clean.new_tensor(lengths)
        noisy = (clean + sigma * torch.randn_like(clean)) % box
        types = torch.tensor(np.asarray(info["numbers"]) == 14, device=device).long()
        pred = model(types, graph(noisy, lengths, args.cutoff), PHASES.index(phase), sigma, lengths)
        target = wrapped_target(noisy, clean, box, sigma)
        return (pred - target).square().mean(), target.square().mean()

    for step in range(completed + 1, args.updates + 1):
        if STOP or time.monotonic() >= until:
            save()
            return 75
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        # Equal phase weight regardless of atom count or number of frames.
        # Each step averages --batch-size independently noised (frame, sigma)
        # samples per phase before backward: a single sample's gradient is a
        # very high-variance estimate of the denoising-score-matching
        # objective (the noise is redrawn fresh every call), which was
        # confirmed to stall training indefinitely at some sigma scales even
        # with a 50x higher learning rate; averaging several samples first
        # is the standard mitigation and does not change what the loss
        # estimates, only its variance.
        for phase in PHASES:
            total = 0.
            for _ in range(args.batch_size):
                frame = int(np.random.choice(meta["phases"][phase]["train_ids"]))
                sigma = math.exp(np.random.uniform(math.log(args.sigma_min), math.log(sigma_max)))
                loss, _ = loss_for(phase, frame, sigma)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; last saved checkpoint is preserved")
                total = total + loss
            total = total / args.batch_size
            (total / 2).backward()
            losses[phase] = float(total.detach())
        nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        completed = step
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            state = rng_state()
            torch.manual_seed(args.seed + 10000)
            model.eval()
            metrics = {}
            with torch.no_grad():
                for phase in PHASES:
                    metrics[phase] = {}
                    for label, sigma in (("small", args.sigma_min), ("local", min(0.3, sigma_max)),
                                         ("middle", math.sqrt(args.sigma_min * sigma_max)), ("terminal", sigma_max)):
                        values = [loss_for(phase, frame, sigma) for frame in meta["phases"][phase]["valid_ids"][:4]]
                        metrics[phase][label] = dict(mse=float(torch.stack([v[0] for v in values]).mean()),
                            zero_predictor_mse=float(torch.stack([v[1] for v in values]).mean()))
            restore_rng(state)
            row = dict(step=step, train=losses, validation=metrics)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    return 0


@torch.no_grad()
def generate(args):
    ck = load_pt(args.checkpoint)
    if ck.get("format") != FORMAT:
        raise ValueError("A test40 checkpoint is required")
    device = device_for(args.device)
    config = ck["settings"]["architecture"]
    model = PhaseScore(**config).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    info = ck["metadata"]["phases"][args.phase]
    # A generated trajectory needs one fixed target cell; with per-frame
    # lengths (e.g. NPT "cell breathing" in the source MD) that means picking
    # one reference frame. Default to the first training frame; --frame
    # overrides. The choice is recorded in settings/generation.json below.
    cell_frame = args.frame if args.frame is not None else info["train_ids"][0]
    lengths = info["lengths"][cell_frame]
    box = torch.tensor(lengths, dtype=torch.float32, device=device)
    types = torch.tensor(np.asarray(info["numbers"]) == 14, device=device).long()
    levels = np.geomspace(ck["settings"]["sigma_max"], ck["settings"]["sigma_min"], args.steps + 1)
    settings = dict(checkpoint_sha256=digest(args.checkpoint), phase=args.phase, steps=args.steps,
                    seed=args.seed, device=args.device, init="uniform_periodic", cell_frame=cell_frame)
    out = output_dir(args.output, args.resume)
    completed = 0
    if args.resume:
        state = load_pt(out / "restart.pt")
        if state["settings"] != settings:
            raise ValueError("Generation settings or checkpoint changed")
        pos, completed = state["positions"].to(device), state["step"]
        restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(out / "positions.npy", mode="r+")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pos = torch.rand((len(types), 3), device=device) * box
        trajectory = np.lib.format.open_memmap(out / "positions.npy", mode="w+", dtype="float32",
                                              shape=(args.steps + 1, len(types), 3))
        trajectory[0] = pos.cpu().numpy()
    until = deadline(args)

    def save():
        trajectory.flush()
        save_pt(out / "restart.pt", dict(settings=settings, positions=pos.cpu(), step=completed, rng=rng_state()))
        save_json(out / "generation.json", dict(format=FORMAT, settings=settings, step=completed,
            valid_frames=completed + 1, complete=completed == args.steps, numbers=info["numbers"],
            lengths=lengths, is_equilibrium_trajectory=False))

    for step in range(completed, args.steps):
        if STOP or time.monotonic() >= until:
            save()
            return 75
        sigma = float(levels[step])
        dv = float(levels[step]**2 - levels[step + 1]**2)
        pred = model(types, graph(pos, lengths, config["cutoff"]),
                     PHASES.index(args.phase), sigma, lengths)
        # Euler reverse VE-SDE: score = -prediction/sigma; diffusion sqrt(delta variance).
        pos = (pos - dv / sigma * pred + math.sqrt(dv) * torch.randn_like(pos)) % box
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite sample")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"{args.phase}: {completed}/{args.steps}, sigma={sigma:.4g}", flush=True)
    save()
    # No repeated zero-noise polishing: final sample retains sigma_min smoothing.
    atoms = Atoms(numbers=info["numbers"], positions=pos.cpu().numpy(), cell=np.diag(lengths), pbc=True)
    if info["representation"] == "tetrahedron":
        atoms.set_masses(info["bead_masses_amu"])
    atoms.info.update(phase_condition=args.phase, sigma_min_A=ck["settings"]["sigma_min"],
                      is_equilibrium_sample=False, representation=info["representation"])
    ase.io.write(out / "final.extxyz", atoms)
    if info["representation"] == "tetrahedron":
        # LAMMPS masses belong to types, so preserve distinct template bead masses.
        masses = atoms.get_masses()
        unique, ids = np.unique(np.round(masses, 8), return_inverse=True)
        with (out / "final.data").open("w") as stream:
            stream.write("test40 SiO4 CG positions; structure generator, no force field supplied\n\n")
            stream.write(f"{len(atoms)} atoms\n{len(unique)} atom types\n\n")
            for side, axis in zip(lengths, "xyz"):
                stream.write(f"0 {side:.12g} {axis}lo {axis}hi\n")
            stream.write("\nMasses\n\n")
            for k, mass in enumerate(unique, 1):
                stream.write(f"{k} {mass:.8f}\n")
            stream.write("\nAtoms # atomic\n\n")
            for k, (typ, xyz) in enumerate(zip(ids, atoms.positions), 1):
                stream.write(f"{k} {typ + 1} " + " ".join(f"{x:.10f}" for x in xyz) + "\n")
    else:
        ase.io.write(out / "final.data", atoms, format="lammps-data", atom_style="atomic", masses=True)
    return 0


def structural_stats(atoms, bond_cutoff=2.2):
    """Actual distance-based coordination; never force four nearest O neighbors."""
    lengths = lengths_of(atoms.cell)
    z = atoms.numbers
    i, j, d = primitive_neighbor_list("ijD", pbc=atoms.pbc, cell=atoms.cell, positions=atoms.positions,
                                     cutoff=min(5.0, min(lengths) / 2 - 1e-6))
    r = np.linalg.norm(d, axis=1)
    cross = (z[i] != z[j]) & (r < bond_cutoff)
    degree = np.bincount(i[cross], minlength=len(z))
    bonds = r[cross & (z[i] == 14)]
    angles = []
    for center in np.flatnonzero(z == 14):
        vec = d[cross & (i == center)]
        norms = np.linalg.norm(vec, axis=1)
        vec = vec[norms > 1e-8] / norms[norms > 1e-8, None]
        for a in range(len(vec)):
            for b in range(a + 1, len(vec)):
                angles.append(float(np.degrees(np.arccos(np.clip(vec[a] @ vec[b], -1, 1)))))
    # Reciprocal-lattice grid gives a direction-resolved order diagnostic, not powder XRD.
    grid = np.array([(a, b, c) for a in range(-4, 5) for b in range(-4, 5)
                     for c in range(-4, 5) if (a, b, c) != (0, 0, 0)])
    wave = 2 * np.pi * grid / lengths
    si = atoms.positions[z == 14]
    sq = np.abs(np.exp(1j * (si @ wave.T)).sum(0)) ** 2 / len(si)
    bins = np.linspace(0, min(5., min(lengths) / 2 - 1e-6), 101)
    pair_hist = {}
    for name, a, b in (("SiSi", 14, 14), ("SiO", 14, 8), ("OO", 8, 8)):
        counts = np.histogram(r[(z[i] == a) & (z[j] == b)], bins)[0]
        pairs = np.sum(z == a) * (np.sum(z == b) - int(a == b))
        shell = 4 * np.pi / 3 * np.diff(bins ** 3)
        pair_hist[name] = (counts * np.prod(lengths) / (pairs * shell)).tolist()
    def minimum(a):
        values = r[(z[i] == a) & (z[j] == a)]
        return float(values.min()) if len(values) else None
    def summary(values):
        return dict(count=len(values), mean=float(np.mean(values)) if len(values) else None,
                    std=float(np.std(values)) if len(values) else None)
    return dict(si_fourfold_fraction=float(np.mean(degree[z == 14] == 4)),
        o_twofold_fraction=float(np.mean(degree[z == 8] == 2)),
        si_coordination=np.bincount(degree[z == 14]).tolist(), o_coordination=np.bincount(degree[z == 8]).tolist(),
        min_SiSi_A=minimum(14), min_OO_A=minimum(8), bonds_A=summary(bonds), angles_deg=summary(angles),
        bond_hist=np.histogram(bonds, np.linspace(0, 3, 121))[0].tolist(),
        angle_hist=np.histogram(angles, np.linspace(0, 180, 91))[0].tolist(),
        rdf_r_A=((bins[:-1] + bins[1:]) / 2).tolist(), rdf=pair_hist,
        reciprocal_indices=grid.tolist(), si_structure_factor=sq.tolist(),
        max_si_structure_factor=float(sq.max()), bond_cutoff_A=bond_cutoff)


def cg_stats(atoms, cutoff=4.0):
    """Bead-level network metrics; no invented internal Si-O distances/angles."""
    box = lengths_of(atoms.cell)
    radius = min(8.0, min(box) / 2 - 1e-6)
    if cutoff >= radius:
        raise ValueError("CG coordination cutoff must be smaller than the RDF radius / half-box")
    i, j, d = primitive_neighbor_list("ijD", pbc=atoms.pbc, cell=atoms.cell,
                                     positions=atoms.positions, cutoff=radius)
    r = np.linalg.norm(d, axis=1)
    degree = np.bincount(i[r < cutoff], minlength=len(atoms))
    bins = np.linspace(0, radius, 161)
    shell = 4 * np.pi / 3 * np.diff(bins**3)
    counts = np.histogram(r, bins)[0]
    rdf = counts * np.prod(box) / (len(atoms) * (len(atoms) - 1) * shell)
    grid = np.array([(a, b, c) for a in range(-4, 5) for b in range(-4, 5)
                     for c in range(-4, 5) if (a, b, c) != (0, 0, 0)])
    sq = np.abs(np.exp(2j * np.pi * ((atoms.positions / box) @ grid.T)).sum(0))**2 / len(atoms)
    return dict(beads=len(atoms), four_neighbor_fraction=float(np.mean(degree == 4)),
        coordination_hist=np.bincount(degree).tolist(), coordination_cutoff_A=cutoff,
        min_bead_distance_A=float(r.min()) if len(r) else None,
        rdf_r_A=((bins[:-1] + bins[1:]) / 2).tolist(), rdf=rdf.tolist(),
        reciprocal_indices=grid.tolist(), structure_factor=sq.tolist(),
        max_structure_factor=float(sq.max()))


def evaluate(args):
    arrays, meta = load_dataset(args.dataset)
    info = meta["phases"][args.phase]
    sample = ase.io.read(args.sample)
    all_lengths = np.asarray(info["lengths"])
    sample_lengths = lengths_of(sample.cell)
    # Frames may each have their own cell (e.g. NPT "cell breathing"), so
    # check the sample cell falls within the observed per-axis range instead
    # of requiring exact equality to one fixed phase-wide cell.
    in_range = np.all((sample_lengths >= all_lengths.min(0) - 1e-5) & (sample_lengths <= all_lengths.max(0) + 1e-5))
    if not sample.pbc.all() or sorted(sample.numbers) != sorted(info["numbers"]) or not in_range:
        raise ValueError("Sample composition/cell must match the selected phase dataset")
    is_cg = info["representation"] == "tetrahedron"
    stats = cg_stats if is_cg else structural_stats
    cutoff = args.bond_cutoff or (4.0 if is_cg else 2.2)
    refs = [stats(Atoms(numbers=info["numbers"], positions=arrays[args.phase][i],
        cell=np.diag(info["lengths"][i]), pbc=True), cutoff) for i in info["valid_ids"][:args.reference_frames]]
    report = dict(phase=args.phase, validation_mode=info["validation_mode"],
                  representation=info["representation"], sample=stats(sample, cutoff), references=refs,
                  note="No automatic scientific pass/fail. Compare coordination, RDF and reciprocal peaks across independent samples.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    keys = ("four_neighbor_fraction", "min_bead_distance_A", "max_structure_factor") if is_cg else ("si_fourfold_fraction", "o_twofold_fraction", "max_si_structure_factor")
    print(json.dumps({k: report["sample"][k] for k in keys}), flush=True)
    if is_cg:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ref in refs:
            axes[0].plot(ref["rdf_r_A"], ref["rdf"], color="C0", alpha=.4)
        axes[0].plot(report["sample"]["rdf_r_A"], report["sample"]["rdf"], color="C1", label="generated")
        axes[0].plot([], [], color="C0", label="reference")
        axes[0].set(xlabel="Bead distance (Angstrom)", ylabel="g(r)")
        axes[0].legend()
        wave = 2 * np.pi * np.asarray(refs[0]["reciprocal_indices"]) / sample_lengths
        magnitude = np.linalg.norm(wave, axis=1)
        axes[1].scatter(magnitude, np.mean([r["structure_factor"] for r in refs], axis=0), s=8, alpha=.4, label="reference")
        axes[1].scatter(magnitude, report["sample"]["structure_factor"], s=8, alpha=.4, label="generated")
        axes[1].set(xlabel="|k| (1/Angstrom)", ylabel="S(k), discrete reciprocal vectors")
        axes[1].legend()
        fig.suptitle(f"test40 SiO4 beads: {args.phase}")
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(fig)
    return 0


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return value


def count(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for phase in PHASES:
        prep.add_argument("--" + phase, type=Path, nargs="+", required=True)
        prep.add_argument("--repeat-" + phase, type=count, nargs=3, default=(1, 1, 1))
        prep.add_argument("--" + phase + "-input-format", dest=phase + "_input_format", default=None,
                           help="ASE format for --" + phase + ", e.g. lammps-data or lammps-dump-text; auto-detected if omitted")
        prep.add_argument("--" + phase + "-lammps-types", dest=phase + "_lammps_types", type=int, nargs="+",
                           help="Atomic numbers for LAMMPS types 1,2,... in --" + phase)
        prep.add_argument("--" + phase + "-index", dest=phase + "_index", default=":",
                           help="ASE frame slice for --" + phase + ", e.g. ::10 (per-phase: a single-frame quench and a long trajectory usually need different slices)")
    prep.add_argument("--validation-fraction", type=positive, default=0.1)
    prep.add_argument("--split-gap", type=int, default=0, help="Discard this many frames before held-out tail")
    prep.add_argument("--allow-single-reference", action="store_true")
    prep.add_argument("--representation", choices=("tetrahedron", "atomic"), default="tetrahedron")
    prep.add_argument("--mapping-cutoff", type=positive, default=2.2, help="Si-O distance defining shared SiO4 tetrahedra, Angstrom")
    prep.add_argument("--allow-sharing-defects", action="store_true", help="Allow O shared by 1 or >2 tetrahedra; divide its mass by actual sharing count. Si still must have four O.")
    prep.add_argument("--allow-topology-drift", action="store_true", help="Allow frames whose Si-O connectivity differs (e.g. independently quenched glass replicas). Training is unaffected (each frame's beads are computed from its own topology); only the single stored bead_masses_amu (generate()'s output masses) reflects one frame, not every frame exactly.")
    prep.add_argument("--output", type=Path, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", type=Path, required=True)
    tr.add_argument("--updates", type=count, default=30000)
    tr.add_argument("--batch-size", type=count, default=8, help="(frame, sigma) samples averaged per phase before each backward")
    tr.add_argument("--width", type=count, default=64)
    tr.add_argument("--layers", type=count, default=3)
    tr.add_argument("--cutoff", type=positive, default=5.)
    tr.add_argument("--sigma-min", type=positive, default=0.03)
    tr.add_argument("--sigma-max", type=positive)
    tr.add_argument("--learning-rate", type=positive, default=2e-4)
    tr.add_argument("--log-every", type=count, default=100)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", type=Path, required=True)
    gen.add_argument("--phase", choices=PHASES, required=True)
    gen.add_argument("--steps", type=count, default=1000)
    gen.add_argument("--frame", type=int, help="Dataset frame index to use as the fixed generation cell; default is the first training frame")
    for q in (tr, gen):
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        q.add_argument("--seed", type=int, default=1337)
        q.add_argument("--checkpoint-every", type=count, default=100)
        q.add_argument("--time-budget-hours", type=float, default=19.5, help="0 disables deadline")
        q.add_argument("--resume", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", type=Path, required=True)
    ev.add_argument("--phase", choices=PHASES, required=True)
    ev.add_argument("--sample", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--reference-frames", type=count, default=4)
    ev.add_argument("--bond-cutoff", type=positive, help="Coordination cutoff: default 4.0 A for CG, 2.2 A for atomic")
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare" and (args.validation_fraction >= 1 or args.split_gap < 0):
        raise ValueError("validation-fraction must be <1 and split-gap nonnegative")
    if hasattr(args, "time_budget_hours") and (not math.isfinite(args.time_budget_hours) or args.time_budget_hours < 0):
        raise ValueError("time-budget-hours must be finite and nonnegative")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import hashlib
import math
from pathlib import Path

import h5py as h5
import numpy as np

from vimes.preprocess import (
    FRAMES_PER_PHASE,
    INTERPOLATED_FRAMES_FOR_JUMP,
    JUMP_THRESHOLD_ABS,
    JUMP_THRESHOLD_RATIO,
    MT_PADDING_FRAMES,
    detect_large_jump,
    detect_mt_starts,
    detect_phases_indices,
    get_stellar_types,
    interp,
    load_hdf5_and_mask,
    make_event_string,
    sample_indices,
)
from vimes.temp_to_color import (
    interp_from_hdf5,
    load_hdf5_temperatures,
    temperature_to_rgb,
)

FRAME_KEYS = (
    "Time",
    "SemiMajorAxis",
    "Eccentricity",
    "Radius(1)",
    "Radius(2)",
    "Mass(1)",
    "Mass(2)",
)
COMPARED_KEYS = (
    *FRAME_KEYS,
    "stypeName1",
    "stypeName2",
    "eventString",
    "Teff1",
    "Teff2",
)
JUMP_KINDS = ("jump", "phase_jump")
SN_WINDOW_FRAMES = 50

CONTINUOUS_COLUMNS = (
    "MassTransferRateDonor",
    "Beta",
    "RocheLobe(1)",
    "RocheLobe(2)",
    "Mass_Env(1)",
    "Mass_Env(2)",
    "Mass_Core(1)",
    "Mass_Core(2)",
    "Omega(1)",
    "Omega(2)",
    "Omega_Break(1)",
    "Omega_Break(2)",
    "Tau_Dynamical(1)",
    "Tau_Dynamical(2)",
    "Tau_Thermal(1)",
    "Tau_Thermal(2)",
    "Tau_Radial(1)",
    "Tau_Radial(2)",
    "Luminosity(1)",
    "Luminosity(2)",
    "Mdot(1)",
    "Mdot(2)",
    "dmMT(1)",
    "dmMT(2)",
    "dmWinds(1)",
    "dmWinds(2)",
    "Ang_Momentum(1)",
    "Ang_Momentum(2)",
    "Ang_Momentum_Total",
    "Pulsar_Spin_Period(1)",
    "Pulsar_Spin_Period(2)",
    "Pulsar_Mag_Field(1)",
    "Pulsar_Mag_Field(2)",
)
CODE_COLUMNS = (
    "MT_History",
    "MassTransferTimescale",
    "Stellar_Type(1)",
    "Stellar_Type(2)",
    "Unbound",
    "Dominant_Mass_Loss_Rate(1)",
    "Dominant_Mass_Loss_Rate(2)",
)
EXPECTED_UNITS = {
    "Time": "Myr",
    "dT": "Myr",
    "Record_Type": "-",
    "SemiMajorAxis": "Rsol",
    "Eccentricity": "-",
    "Mass(1)": "Msol",
    "Mass(2)": "Msol",
    "Radius(1)": "Rsol",
    "Radius(2)": "Rsol",
    "Teff(1)": "K",
    "Teff(2)": "K",
    "Luminosity(1)": "Lsol",
    "Luminosity(2)": "Lsol",
    "Stellar_Type(1)": "-",
    "Stellar_Type(2)": "-",
    "MT_History": "-",
    "MassTransferTimescale": "-",
    "MassTransferRateDonor": "Msol/Myr",
    "Beta": "-",
    "RocheLobe(1)": "Rsol",
    "RocheLobe(2)": "Rsol",
    "Mass_Env(1)": "Msol",
    "Mass_Env(2)": "Msol",
    "Mass_Core(1)": "Msol",
    "Mass_Core(2)": "Msol",
    "Mass_He_Core(1)": "Msol",
    "Mass_He_Core(2)": "Msol",
    "Mass_CO_Core(1)": "Msol",
    "Mass_CO_Core(2)": "Msol",
    "Omega(1)": "Hz",
    "Omega(2)": "Hz",
    "Omega_Break(1)": "Hz",
    "Omega_Break(2)": "Hz",
    "Tau_Dynamical(1)": "Myr",
    "Tau_Dynamical(2)": "Myr",
    "Tau_Thermal(1)": "Myr",
    "Tau_Thermal(2)": "Myr",
    "Tau_Radial(1)": "Myr",
    "Tau_Radial(2)": "Myr",
    "Unbound": "State",
    "Mdot(1)": "Msol yr^-1",
    "Mdot(2)": "Msol yr^-1",
    "dmMT(1)": "Msol",
    "dmMT(2)": "Msol",
    "dmWinds(1)": "Msol",
    "dmWinds(2)": "Msol",
    "Ang_Momentum(1)": "Msol AU^2 yr^-1",
    "Ang_Momentum(2)": "Msol AU^2 yr^-1",
    "Ang_Momentum_Total": "Msol AU^2 yr^-1",
    "Pulsar_Spin_Period(1)": "s",
    "Pulsar_Spin_Period(2)": "s",
    "Pulsar_Mag_Field(1)": "Gauss",
    "Pulsar_Mag_Field(2)": "Gauss",
    "Dominant_Mass_Loss_Rate(1)": "-",
    "Dominant_Mass_Loss_Rate(2)": "-",
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(kind, src, trigger=-1):
    return {"kind": kind, "src": src, "trigger": trigger}


def sample_src(rows, pos):
    if abs(pos - round(pos)) < 1e-9:
        return int(rows[int(pos)]), int(rows[int(pos)])
    return int(rows[math.floor(pos)]), int(rows[math.ceil(pos)])


def bracket(first, last):
    return first["src"][0], last["src"][1]


def is_jump(a, b):
    sep_a = a["SemiMajorAxis"] * (1 + a["Eccentricity"])
    sep_b = b["SemiMajorAxis"] * (1 + b["Eccentricity"])
    return (
        detect_large_jump(a["Radius(1)"], b["Radius(1)"])
        or detect_large_jump(a["Radius(2)"], b["Radius(2)"])
        or detect_large_jump(sep_a, sep_b)
    )


def jump_frames(a, b):
    inserted = []
    for j in range(1, INTERPOLATED_FRAMES_FOR_JUMP + 1):
        alpha = j / (INTERPOLATED_FRAMES_FOR_JUMP + 1)
        inter = {k: (1 - alpha) * a[k] + alpha * b[k] for k in FRAME_KEYS}
        inter["stypeName1"] = a["stypeName1"]
        inter["stypeName2"] = a["stypeName2"]
        inter["eventString"] = a["eventString"]
        inserted.append(inter)
    return inserted


def replay_temperatures(hdf5_file, frames):
    hdf5_time, teff1, teff2 = load_hdf5_temperatures(hdf5_file)
    for f in frames:
        t = f["Time"]
        temperature1 = interp_from_hdf5(t, hdf5_time, teff1)
        temperature2 = interp_from_hdf5(t, hdf5_time, teff2)
        f["Teff1"] = float(temperature1)
        f["Teff2"] = float(temperature2)
        f["RGB1"] = temperature_to_rgb(temperature1)
        f["RGB2"] = temperature_to_rgb(temperature2)


def replay_frames(hdf5_file):
    data = load_hdf5_and_mask(hdf5_file)
    with h5.File(hdf5_file, "r") as handle:
        record_type = handle["Record_Type"][()]
        mt_timescale = handle["MassTransferTimescale"][()]
    rows = np.flatnonzero((record_type == 4) | (mt_timescale == 3))
    type_map = get_stellar_types()
    phases = detect_phases_indices(data["Stellar_Type(1)"], data["Stellar_Type(2)"])
    mt_starts = detect_mt_starts(data["MT_History"].astype(int))

    frames = []
    sources = []
    for start, end in phases:
        sampled = []
        sampled_sources = []
        for pos in sample_indices(start, end, FRAMES_PER_PHASE):
            f = {k: interp(data[k], pos) for k in FRAME_KEYS}
            f["stypeName1"] = type_map(round(interp(data["Stellar_Type(1)"], pos)))
            f["stypeName2"] = type_map(round(interp(data["Stellar_Type(2)"], pos)))
            f["eventString"] = make_event_string(round(pos), data, type_map)
            sampled.append(f)
            sampled_sources.append(provenance("sample", sample_src(rows, pos)))

        enhanced = []
        enhanced_sources = []
        for i in range(len(sampled) - 1):
            a, b = sampled[i], sampled[i + 1]
            enhanced.append(a)
            enhanced_sources.append(sampled_sources[i])
            if is_jump(a, b):
                src = bracket(sampled_sources[i], sampled_sources[i + 1])
                for inter in jump_frames(a, b):
                    enhanced.append(inter)
                    enhanced_sources.append(provenance("jump", src))
        enhanced.append(sampled[-1])
        enhanced_sources.append(sampled_sources[-1])

        mt_frames = []
        mt_sources = []
        last_mt_frame = -999
        for i, f in enumerate(enhanced):
            mt_frames.append(f)
            mt_sources.append(enhanced_sources[i])

            data_idx = round(start + (end - start) * (i / max(1, len(enhanced) - 1)))

            is_ce_event = (
                data_idx < len(data["MassTransferTimescale"])
                and int(data["MassTransferTimescale"][data_idx]) == 3
            )

            if is_ce_event:
                event_string, pad_kind = "Common Envelope", "ce_pad"
            elif data_idx in mt_starts:
                event_string, pad_kind = "Mass Transfer", "mt_pad"
            else:
                continue

            if i - last_mt_frame < 25:
                continue

            copied = enhanced_sources[i]["src"]
            trigger = int(rows[data_idx])
            for _ in range(MT_PADDING_FRAMES):
                g = f.copy()
                g["eventString"] = event_string
                mt_frames.append(g)
                mt_sources.append(provenance(pad_kind, copied, trigger))

            last_mt_frame = i

        if frames and is_jump(frames[-1], mt_frames[0]):
            src = bracket(sources[-1], mt_sources[0])
            trigger = sources[-1]["trigger"]
            for inter in jump_frames(frames[-1], mt_frames[0]):
                frames.append(inter)
                sources.append(provenance("phase_jump", src, trigger))

        frames.extend(mt_frames)
        sources.extend(mt_sources)

    replay_temperatures(hdf5_file, frames)
    return frames, sources


def mismatch(i, key, found, replayed):
    return f"frame {i}: {key} is {found!r} in the npz but {replayed!r} in the replay"


def check_replay(frames, replayed):
    for i, (frame, expected) in enumerate(zip(frames, replayed, strict=False)):
        for key in COMPARED_KEYS:
            if frame.get(key) != expected[key]:
                raise ValueError(mismatch(i, key, frame.get(key), expected[key]))
        for key in ("RGB1", "RGB2"):
            if not np.array_equal(frame.get(key), expected[key]):
                raise ValueError(mismatch(i, key, frame.get(key), expected[key]))
    if len(frames) != len(replayed):
        first = min(len(frames), len(replayed))
        raise ValueError(
            f"frame {first}: npz has {len(frames)} frames, replay has {len(replayed)}"
        )


def check_units(handle, names):
    for name in names:
        expected = EXPECTED_UNITS[name]
        found = handle[name].attrs["units"].decode()
        if found != expected:
            raise ValueError(f"{name}: expected units {expected!r}, found {found!r}")


def load_type4_columns(hdf5_file, names):
    with h5.File(hdf5_file, "r") as handle:
        check_units(handle, ["Time", *names])
        mask = handle["Record_Type"][()] == 4
        time = handle["Time"][()][mask]
        columns = {name: handle[name][()][mask] for name in names}
    return time, columns


def continuous_value(t, time, values):
    return float(interp_from_hdf5(t, time, values))


def held_value(t, time, values):
    i = int(np.searchsorted(time, t, side="right")) - 1
    return int(values[max(i, 0)])


def endpoint(table, row):
    values = {name: column[row].item() for name, column in table.items()}
    return {"row": row, **values}


def make_event(table, kind, star, rows, roles):
    endpoints = {role: endpoint(table, rows[role]) for role in roles if role in rows}
    return {
        "kind": kind,
        "star": star,
        "rows": rows,
        "endpoints": endpoints,
        "runs": [],
    }


def find_events(hdf5_file):
    with h5.File(hdf5_file, "r") as handle:
        check_units(handle, EXPECTED_UNITS)
        table = {name: handle[name][()] for name in EXPECTED_UNITS}
    record_type = table["Record_Type"]
    mt_history = table["MT_History"]
    mt_timescale = table["MassTransferTimescale"]
    types = {1: table["Stellar_Type(1)"], 2: table["Stellar_Type(2)"]}
    t4 = np.flatnonzero(record_type == 4).tolist()
    stable = [int(mt_history[r]) in (1, 2) and int(mt_timescale[r]) != 3 for r in t4]

    events = []
    for k, row in enumerate(t4):
        pre = t4[max(k - 1, 0)]
        if stable[k] and (k == 0 or not stable[k - 1]):
            last = k
            while last + 1 < len(t4) and stable[last + 1]:
                last += 1
            rows = {"pre": pre, "start": row, "post": t4[last]}
            star = int(mt_history[row])
            events.append(make_event(table, "MT", star, rows, ("pre", "post")))
        if int(mt_timescale[row]) == 3:
            step = range(pre + 1, row)
            rows = {"pre": pre}
            type_change = [r for r in step if record_type[r] == 6]
            if type_change:
                rows["typeChange"] = type_change[0]
            rows["post"] = next((r for r in step if record_type[r] == 12), row)
            rows["close"] = row
            rows["next"] = t4[k + 1] if k + 1 < len(t4) else len(record_type)
            star = {3: 1, 4: 2}[int(mt_history[row])]
            roles = ("pre", "typeChange", "post")
            events.append(make_event(table, "CE", star, rows, roles))

    for s in np.flatnonzero(record_type == 13).tolist():
        pre = max(r for r in t4 if r < s)
        changed = [n for n in (1, 2) if types[n][s] != types[n][pre]]
        star = next(n for n in changed if types[n][s] in (13, 14))
        kept = np.flatnonzero(types[star][pre:s] == types[star][pre])
        collapse = pre + int(kept[-1])
        rows = {"pre": pre, "collapse": collapse, "preSN": s - 1, "post": s}
        roles = ("pre", "collapse", "preSN", "post")
        events.append(make_event(table, "SN", star, rows, roles))

    events.sort(key=lambda event: event["rows"]["pre"])
    return [{"id": i, **event} for i, event in enumerate(events)]


def contiguous_runs(indices):
    runs = []
    for i in indices:
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def set_episode(frame, ep, part, index, count):
    frame["x3d.ep"] = ep
    frame["x3d.epKind"] = part
    frame["x3d.epIndex"] = index
    frame["x3d.epCount"] = count


def claim(frames, event, part, indices):
    free = [i for i in indices if frames[i]["x3d.ep"] == -1]
    for run in contiguous_runs(free):
        event["runs"].append([run[0], run[-1], part])
        for k, i in enumerate(run):
            set_episode(frames[i], event["id"], part, k, len(run))
            for role, values in event["endpoints"].items():
                frames[i]["x3d.ep" + role[0].upper() + role[1:]] = dict(values)


def mt_pads(sources, start):
    return [
        i
        for i, source in enumerate(sources)
        if source["kind"] == "mt_pad" and source["trigger"] == start
    ]


def ce_frames(frames, sources, rows):
    indices = []
    for i, frame in enumerate(frames):
        inside = rows["pre"] < sources[i]["trigger"] < rows["next"]
        if frame["eventString"] == "Common Envelope" and inside:
            indices.append(i)
    return indices


def covering_jumps(sources, jump_runs, first, last):
    covering = []
    for run in jump_runs:
        low, high = sources[run[0]]["src"]
        if low <= first and last <= high:
            covering.extend(run)
    return covering


def sn_window(sources, explosion):
    later = (i for i, s in enumerate(sources) if s["src"][0] > explosion)
    first = next(later, len(sources))
    return range(first, min(first + SN_WINDOW_FRAMES, len(sources)))


def attach_episodes(frames, sources, events):
    for frame in frames:
        set_episode(frame, -1, "", -1, 0)
    jumps = [i for i, s in enumerate(sources) if s["kind"] in JUMP_KINDS]
    jump_runs = contiguous_runs(jumps)
    for event in events:
        kind = event["kind"]
        rows = event["rows"]
        if kind == "MT":
            claim(frames, event, "mt", mt_pads(sources, rows["start"]))
        elif kind == "CE":
            claim(frames, event, "ce", ce_frames(frames, sources, rows))
        last = rows["start"] if kind == "MT" else rows["post"]
        covering = covering_jumps(sources, jump_runs, rows["pre"], last)
        claim(frames, event, kind.lower() + "_jump", covering)
        if kind == "SN":
            claim(frames, event, "sn_window", sn_window(sources, rows["post"]))
        event["runs"].sort()


def make_meta(hdf5_file, frames_file, npz_sha256, events):
    with h5.File(hdf5_file, "r") as handle:
        seed = str(handle["SEED"][0])
    return {
        "h5": {"name": Path(hdf5_file).name, "sha256": sha256_of(hdf5_file)},
        "npz": {"name": Path(frames_file).name, "sha256": npz_sha256},
        "seed": seed,
        "constants": {
            "FRAMES_PER_PHASE": FRAMES_PER_PHASE,
            "INTERPOLATED_FRAMES_FOR_JUMP": INTERPOLATED_FRAMES_FOR_JUMP,
            "MT_PADDING_FRAMES": MT_PADDING_FRAMES,
            "JUMP_THRESHOLD_RATIO": JUMP_THRESHOLD_RATIO,
            "JUMP_THRESHOLD_ABS": JUMP_THRESHOLD_ABS,
        },
        "units": dict(EXPECTED_UNITS),
        "continuous": list(CONTINUOUS_COLUMNS),
        "codes": list(CODE_COLUMNS),
        "events": events,
        "unattached": [event["id"] for event in events if not event["runs"]],
    }


def add_physics_columns(hdf5_file, frames_file, output_frames_file):
    npz_sha256 = sha256_of(frames_file)
    frames = np.load(frames_file, allow_pickle=True)["frames"].tolist()
    if any(key.startswith("x3d.") for frame in frames for key in frame):
        raise ValueError(f"{frames_file} already has x3d. keys")

    replayed, sources = replay_frames(hdf5_file)
    check_replay(frames, replayed)

    time, columns = load_type4_columns(hdf5_file, CONTINUOUS_COLUMNS + CODE_COLUMNS)
    for frame, source in zip(frames, sources, strict=True):
        t = frame["Time"]
        for name in CONTINUOUS_COLUMNS:
            frame["x3d." + name] = continuous_value(t, time, columns[name])
        for name in CODE_COLUMNS:
            frame["x3d." + name] = held_value(t, time, columns[name])
        frame["x3d.kind"] = source["kind"]
        frame["x3d.src"] = list(source["src"])

    events = find_events(hdf5_file)
    attach_episodes(frames, sources, events)
    frames[0]["x3d.meta"] = make_meta(hdf5_file, frames_file, npz_sha256, events)

    np.savez_compressed(output_frames_file, frames=frames)
    print(f"Saved {len(frames)} frames → {output_frames_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add x3d.* keys for the 3D view to a VIMES frames file."
    )
    parser.add_argument("hdf5", type=Path, help="Path to the input HDF5 file.")
    parser.add_argument("frames", type=Path, help="Path to the input frames file.")
    parser.add_argument(
        "out",
        type=Path,
        nargs="?",
        help="Path to the output frames file (default: <frames>_3d.npz).",
    )
    parser.add_argument(
        "--in-place", action="store_true", help="Write back to the input frames file."
    )
    args = parser.parse_args()

    if args.in_place and args.out is not None:
        parser.error("--in-place cannot be combined with an output path")
    for path in (args.hdf5, args.frames):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

    out = args.out or args.frames.with_name(args.frames.stem + "_3d.npz")
    add_physics_columns(args.hdf5, args.frames, args.frames if args.in_place else out)


if __name__ == "__main__":
    main()

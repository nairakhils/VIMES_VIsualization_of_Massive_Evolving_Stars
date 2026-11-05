import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import strix
from PIL import Image
from tulips.blackbody import blackbody_color

from vimes.animate import BACKGROUND_PATH, STELLAR_IMG_DIR
from vimes.temp_to_color import temperature_to_rgb

KINDS = {
    "NS": "NS",
    "BH": "BH",
    "HeWD": "WD",
    "COWD": "WD",
    "ONeWD": "WD",
    "MR": "none",
}
TYPE_NAMES = {
    "MS": "Main sequence",
    "HG": "Hertzsprung gap",
    "FGB": "First giant branch",
    "CHeB": "Core He burning",
    "EAGB": "Early AGB",
    "TPAGB": "Thermally pulsing AGB",
    "HeMS": "Naked He MS",
    "HeHG": "Naked He HG",
    "HeGB": "Naked He GB",
    "HeWD": "He white dwarf",
    "COWD": "CO white dwarf",
    "ONeWD": "ONe white dwarf",
    "NS": "Neutron star",
    "BH": "Black hole",
    "MR": "Massless remnant",
    "CHE": "Chem. homogeneous",
}
STAGES = {"Mass Transfer": "transfer", "Common Envelope": "envelope"}
REMNANTS = ("NS", "BH")
CE_RATE = 1e300
HOLD_WINDOW = 60
ICON_SIZE = 64
SKY = "Background.png"
PALETTE_TEMPERATURES = np.geomspace(1000.0, 2e6, 256)


def column(frames, key):
    return np.array([frame[key] for frame in frames], dtype=float)


def kind_of(stellar_type):
    return KINDS.get(stellar_type, "star")


def unbound_runs(bound):
    runs = []
    n = 0
    while n < len(bound):
        if bound[n]:
            n += 1
            continue
        start = n
        while n + 1 < len(bound) and not bound[n + 1]:
            n += 1
        runs.append((start, n))
        n += 1
    return runs


def explosion_frame(types, start, end):
    for n in range(start, min(end, start + HOLD_WINDOW) + 1):
        if any(names[n] != names[n - 1] and names[n] in REMNANTS for names in types):
            return n
    return start


def hold_frame(a, e, start):
    da = a[start] - a[start - 1]
    de = e[start] - e[start - 1]
    k = start - 1
    while (
        k > 0
        and de != 0
        and start - k < HOLD_WINDOW
        and abs(e[k] - e[k - 1] - de) < 1e-3 * abs(de)
        and abs(a[k] - a[k - 1] - da) < 1e-6 + 1e-3 * abs(da)
    ):
        k -= 1
    return k


def hold_orbits(a, e, masses, types):
    bound = (a > 0) & (e >= 0) & (e < 1) & (masses[0] > 0) & (masses[1] > 0)
    for start, end in unbound_runs(bound):
        if start == 0:
            continue
        explosion = explosion_frame(types, start, end)
        held = hold_frame(a, e, start)
        a[held + 1 : explosion] = a[held]
        e[held + 1 : explosion] = e[held]


def runs_of(episode, part):
    return [(first, last) for first, last, name in episode["runs"] if name == part]


def envelope_temperatures(teffs, episodes):
    for episode in episodes:
        if episode["kind"] != "CE" or not episode["runs"]:
            continue
        pre, post = episode["endpoints"]["pre"], episode["endpoints"]["post"]
        jump = next(iter(runs_of(episode, "ce_jump")), None)
        first = jump[0] if jump else min(start for start, _, _ in episode["runs"])
        last = max(end for _, end, _ in episode["runs"])
        for n in range(first, last + 1):
            w = (
                (n - jump[0] + 1) / (jump[1] - jump[0] + 2)
                if jump and n <= jump[1]
                else 1
            )
            for k, teff in enumerate(teffs):
                before = np.log(pre[f"Teff({k + 1})"])
                teff[n] = np.exp(before + (np.log(post[f"Teff({k + 1})"]) - before) * w)


def stages(frames, episodes, rates):
    stage = [STAGES.get(frame["eventString"], "") for frame in frames]
    for name, kind in (("transfer", "MT"), ("envelope", "CE")):
        for episode in episodes:
            if episode["kind"] == kind:
                for first, last, _ in episode["runs"]:
                    stage[first : last + 1] = [name] * (last - first + 1)
    if rates is not None:
        for n, rate in enumerate(rates):
            if rate > CE_RATE:
                stage[n] = "envelope"
            elif rate > 0 and stage[n] == "":
                stage[n] = "transfer"
    return stage


def span(runs):
    return (
        [min(first for first, _ in runs), max(last for _, last in runs)]
        if runs
        else None
    )


def orbit(endpoint):
    return {
        "time": endpoint["Time"],
        "m1": endpoint["Mass(1)"],
        "m2": endpoint["Mass(2)"],
        "a": endpoint["SemiMajorAxis"],
        "e": endpoint["Eccentricity"],
    }


def stars_state(endpoint):
    return {
        **orbit(endpoint),
        "r1": endpoint["Radius(1)"],
        "r2": endpoint["Radius(2)"],
        "teff1": endpoint["Teff(1)"],
        "teff2": endpoint["Teff(2)"],
    }


def envelope_events(episodes):
    events = []
    for episode in episodes:
        if episode["kind"] != "CE" or not episode["runs"]:
            continue
        during = span(runs_of(episode, "ce_jump")) or span(runs_of(episode, "ce"))
        events.append(
            {
                "kind": "envelope",
                "donor": episode["star"],
                "during": during,
                "until": max(last for _, last, _ in episode["runs"]),
                "start": stars_state(episode["endpoints"]["pre"]),
                "end": stars_state(episode["endpoints"]["post"]),
            }
        )
    return events


def layers(progenitor, k):
    mass = progenitor[f"Mass({k})"]
    helium = progenitor[f"Mass_He_Core({k})"]
    carbon = progenitor[f"Mass_CO_Core({k})"]
    return {"hydrogen": max(mass - helium, 0.0), "helium": max(helium - carbon, 0.0)}


def collapse_events(episodes):
    events = []
    for episode in episodes:
        if episode["kind"] != "SN" or not episode["runs"]:
            continue
        ends = episode["endpoints"]
        events.append(
            {
                "kind": "collapse",
                "star": episode["star"],
                **layers(ends["collapse"], episode["star"]),
                "during": span(runs_of(episode, "sn_jump"))
                or span(runs_of(episode, "sn_window")),
                "until": max(last for _, last, _ in episode["runs"]),
                "start": orbit(ends["collapse"]),
                "end": orbit(ends["post"]),
            }
        )
    return events


def transfer_events(episodes):
    events = []
    for episode in episodes:
        if episode["kind"] != "MT" or not episode["runs"]:
            continue
        during = span(runs_of(episode, "mt_jump")) or span(runs_of(episode, "mt"))
        events.append(
            {
                "kind": "transfer",
                "donor": episode["star"],
                "during": during,
                "until": max(last for _, last, _ in episode["runs"]),
                "start": stars_state(episode["endpoints"]["pre"]),
                "end": stars_state(episode["endpoints"]["post"]),
            }
        )
    return events


def spin_ratios(frames, k):
    omega = column(frames, f"x3d.Omega({k})")
    breakup = column(frames, f"x3d.Omega_Break({k})")
    ratio = np.divide(omega, breakup, out=np.zeros_like(omega), where=breakup > 0)
    return np.where((ratio > 0) & (ratio < 1), ratio, 0.0)


def scene_columns(frames, episodes):
    a = column(frames, "SemiMajorAxis")
    e = column(frames, "Eccentricity")
    masses = [column(frames, "Mass(1)"), column(frames, "Mass(2)")]
    teffs = [column(frames, "Teff1"), column(frames, "Teff2")]
    types = [
        [frame["stypeName1"] for frame in frames],
        [frame["stypeName2"] for frame in frames],
    ]
    hold_orbits(a, e, masses, types)
    envelope_temperatures(teffs, episodes)
    columns = {
        "time": column(frames, "Time"),
        "a": a,
        "e": e,
        "m1": masses[0],
        "m2": masses[1],
        "r1": column(frames, "Radius(1)"),
        "r2": column(frames, "Radius(2)"),
        "teff1": teffs[0],
        "teff2": teffs[1],
        "kind1": [kind_of(name) for name in types[0]],
        "kind2": [kind_of(name) for name in types[1]],
        "type1": types[0],
        "type2": types[1],
        "label": [frame["eventString"] for frame in frames],
    }
    rates = None
    if "x3d.MassTransferRateDonor" in frames[0]:
        rates = column(frames, "x3d.MassTransferRateDonor")
        columns["transfer"] = np.where(
            (rates > 0) & (rates < CE_RATE), rates * 1e-6, 0.0
        )
        for k in (1, 2):
            columns[f"spin{k}"] = spin_ratios(frames, k)
            columns[f"wind{k}"] = column(frames, f"x3d.Mdot({k})")
    columns["stage"] = stages(frames, episodes, rates)
    return columns


def palette():
    return {
        "temperature": PALETTE_TEMPERATURES.tolist(),
        "rgb": [[int(c) for c in temperature_to_rgb(t)] for t in PALETTE_TEMPERATURES],
        "luminance": [float(blackbody_color(t)[1]) for t in PALETTE_TEMPERATURES],
    }


def icon(path):
    with Image.open(path) as image:
        small = image.convert("RGBA").resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS)
    buffer = io.BytesIO()
    small.save(buffer, "PNG")
    return buffer.getvalue()


def type_assets():
    assets = {
        f"{name}.png": icon(STELLAR_IMG_DIR / f"{name}.png") for name in TYPE_NAMES
    }
    types = {
        name: {"name": title, "icon": f"{name}.png"}
        for name, title in TYPE_NAMES.items()
    }
    return assets, types


def convert(frames_path, images="default"):
    frames_path = Path(frames_path)
    content = frames_path.read_bytes()
    frames = np.load(io.BytesIO(content), allow_pickle=True)["frames"].tolist()
    meta = frames[0].get("x3d.meta")
    episodes = meta["events"] if meta else []
    assets, types = type_assets()
    if images == "default":
        assets[SKY] = BACKGROUND_PATH.read_bytes()
    source = {
        "code": "COMPAS via VIMES",
        "frames": frames_path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "h5": meta["h5"] if meta else None,
        "seed": meta["seed"] if meta else None,
    }
    path = strix.write(
        frames_path.with_suffix(".strix"),
        scene_columns(frames, episodes),
        events=transfer_events(episodes)
        + collapse_events(episodes)
        + envelope_events(episodes),
        palette=palette(),
        types=types,
        sky=SKY if images == "default" else None,
        assets=assets,
        source=source,
    )
    print(f"Saved {len(frames)} frames → {path}")
    return path


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Show or render a VIMES run in 3D with strix."
    )
    parser.add_argument("frames", type=Path, help="Path to the input frames file.")
    parser.add_argument(
        "--scaling",
        choices=["log", "linear"],
        default="linear",
        help="The type of scaling to apply (log or linear).",
    )
    parser.add_argument(
        "--camera",
        choices=["fixed", "follow"],
        default="fixed",
        help="Keep the camera still or let it follow the binary.",
    )
    parser.add_argument(
        "--images",
        choices=["tulips", "default"],
        default="default",
        help="The set of images to use (tulips or default).",
    )
    parser.add_argument(
        "--save-mp4", type=str, default=None, help="Save animation to MP4 file"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Run headless (do not open a window)"
    )
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    if not args.frames.exists():
        raise FileNotFoundError(f"{args.frames} not found.")
    scene = convert(args.frames, args.images)
    if args.save_mp4:
        strix.render(scene, args.save_mp4, scale=args.scaling, camera=args.camera)
    if not args.no_display:
        strix.view(scene, scale=args.scaling, camera=args.camera)


if __name__ == "__main__":
    main()

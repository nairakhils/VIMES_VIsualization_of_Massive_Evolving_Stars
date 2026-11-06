import hashlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tulips.blackbody import blackbody_color

from vimes.augment3d import add_physics_columns
from vimes.preprocess import preprocess_to_frames
from vimes.temp_to_color import add_temperatures_and_rgb, temperature_to_rgb

strix = pytest.importorskip("strix")
view3d = pytest.importorskip("vimes.view3d")

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


def forbidden(*args, **kwargs):
    raise AssertionError(f"unexpected call with {args}")


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["view3d", *(str(arg) for arg in args)])
    view3d.main()


def values(frames, key):
    return np.array([frame[key] for frame in frames], dtype=float)


@pytest.fixture(scope="module", params=[0, 1, 2])
def npz_files(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"example{request.param}")
    h5 = EXAMPLES_DIR / f"BSE_Detailed_Output_{request.param}.h5"
    plain = directory / "frames.npz"
    augmented = directory / "frames_3d.npz"
    preprocess_to_frames(h5, plain)
    add_temperatures_and_rgb(h5, plain, plain)
    add_physics_columns(h5, plain, augmented)
    return SimpleNamespace(example=request.param, plain=plain, augmented=augmented)


@pytest.fixture(scope="module", params=["plain", "augmented"])
def converted(request, npz_files):
    npz = getattr(npz_files, request.param)
    header, columns, assets = strix.read(view3d.convert(npz))
    with np.load(npz, allow_pickle=True) as data:
        frames = data["frames"].tolist()
    meta = frames[0].get("x3d.meta")
    return SimpleNamespace(
        kind=request.param,
        example=npz_files.example,
        npz=npz,
        frames=frames,
        episodes=meta["events"] if meta else [],
        header=header,
        columns=columns,
        assets=assets,
    )


@pytest.fixture
def augmented_copy(npz_files, tmp_path):
    path = tmp_path / "frames_3d.npz"
    shutil.copy(npz_files.augmented, path)
    return path


class TestColumns:
    def test_frames_become_samples(self, converted):
        frames, columns = converted.frames, converted.columns
        assert converted.header["samples"] == len(frames)
        for name, key in (("time", "Time"), ("m1", "Mass(1)"), ("m2", "Mass(2)")):
            np.testing.assert_array_equal(columns[name], values(frames, key))
        for name, key in (("r1", "Radius(1)"), ("r2", "Radius(2)")):
            np.testing.assert_array_equal(columns[name], values(frames, key))
        assert columns["type1"] == [frame["stypeName1"] for frame in frames]
        assert columns["label"] == [frame["eventString"] for frame in frames]

    def test_kinds_follow_the_stellar_types(self, converted):
        for name, kind in zip(
            converted.columns["type1"], converted.columns["kind1"], strict=True
        ):
            assert kind == view3d.KINDS.get(name, "star")
        assert set(converted.columns["kind2"]) <= set(strix.KINDS)

    def test_physics_columns_come_with_the_augmented_frames(self, converted):
        optional = {"spin1", "spin2", "wind1", "wind2", "transfer", "period1", "field2"}
        assert (optional <= set(converted.columns)) == (converted.kind == "augmented")
        if converted.kind == "augmented":
            assert all(0 <= s <= 1 for s in converted.columns["spin1"])
            assert all(w >= 0 for w in converted.columns["wind2"])
            for k in (1, 2):
                np.testing.assert_array_equal(
                    converted.columns[f"period{k}"],
                    values(converted.frames, f"x3d.Pulsar_Spin_Period({k})"),
                )
                pairs = zip(
                    converted.columns[f"type{k}"],
                    converted.columns[f"regime{k}"],
                    strict=True,
                )
                seen = {pair for pair in pairs if pair[0] in ("MS", "EAGB")}
                assert seen <= {("MS", "OB"), ("EAGB", "GB")}
            rates = values(converted.frames, "x3d.MassTransferRateDonor")
            inside = (rates > 0) & (rates < view3d.CE_RATE)
            np.testing.assert_allclose(
                converted.columns["transfer"][inside], rates[inside] * 1e-6
            )

    def test_teff_is_vimes_outside_common_envelopes(self, converted):
        envelope = np.zeros(len(converted.frames), bool)
        for episode in converted.episodes:
            if episode["kind"] == "CE" and episode["runs"]:
                first = min(start for start, _, _ in episode["runs"])
                envelope[first : max(end for _, end, _ in episode["runs"]) + 1] = True
        teff = values(converted.frames, "Teff1")
        np.testing.assert_array_equal(
            converted.columns["teff1"][~envelope], teff[~envelope]
        )


class TestVimesArtefacts:
    def test_orbit_held_through_the_interpolated_supernova(self, converted):
        if converted.example != 2:
            pytest.skip("example 2 has the disrupting supernova")
        a, e = converted.columns["a"], converted.columns["e"]
        raw = values(converted.frames, "SemiMajorAxis")
        assert np.all(a[509:560] == a[509])
        assert np.all(e[509:560] == e[509])
        assert a[505] != a[509]
        assert raw[540] != a[540]
        assert a[560] == raw[560] and not (a[560] > 0 and 0 <= e[560] < 1)

    def test_hold_stops_where_the_linear_interpolation_starts(self):
        a = np.array([10.0, 10.0, 10.0, 8.0, 6.0, 4.0, -2.0, -3.0])
        e = np.array([0.1, 0.1, 0.1, 0.4, 0.7, 1.0, 1.3, 1.4])
        masses = [np.ones(8), np.ones(8)]
        types = [["MS"] * 5 + ["NS"] * 3, ["MS"] * 8]
        view3d.hold_orbits(a, e, masses, types)
        np.testing.assert_array_equal(a, [10, 10, 10, 10, 10, 4, -2, -3])
        np.testing.assert_array_equal(e, [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.3, 1.4])

    def test_post_envelope_teff_comes_from_the_endpoints(self, converted):
        ce = [episode for episode in converted.episodes if episode["kind"] == "CE"]
        if not ce:
            pytest.skip("no common envelope")
        episode = ce[0]
        post = episode["endpoints"]["post"]
        last = max(end for _, end, _ in episode["runs"])
        for k in (1, 2):
            assert converted.columns[f"teff{k}"][last] == pytest.approx(
                post[f"Teff({k})"], rel=1e-12
            )

    def test_spin_above_breakup_is_held_at_breakup(self):
        frames = [
            {"x3d.Omega(1)": 2.0, "x3d.Omega_Break(1)": 1.0},
            {"x3d.Omega(1)": 0.5, "x3d.Omega_Break(1)": 1.0},
            {"x3d.Omega(1)": 0.5, "x3d.Omega_Break(1)": 0.0},
        ]
        np.testing.assert_array_equal(view3d.spin_ratios(frames, 1), [1, 0.5, 0])


class TestEvents:
    def test_transfers_follow_the_compas_episodes(self, converted):
        episodes = [e for e in converted.episodes if e["kind"] == "MT" and e["runs"]]
        events = [e for e in converted.header["events"] if e["kind"] == "transfer"]
        assert len(events) == len(episodes)
        for event, episode in zip(events, episodes, strict=True):
            jumps = [run for run in episode["runs"] if run[2] == "mt_jump"]
            assert event["donor"] == episode["star"]
            assert event["during"] == [
                min(r[0] for r in jumps),
                max(r[1] for r in jumps),
            ]
            assert event["until"] == max(r[1] for r in episode["runs"])
            pre = episode["endpoints"]["pre"]
            assert event["start"] == {
                "time": pre["Time"],
                "m1": pre["Mass(1)"],
                "m2": pre["Mass(2)"],
                "a": pre["SemiMajorAxis"],
                "e": pre["Eccentricity"],
                "r1": pre["Radius(1)"],
                "r2": pre["Radius(2)"],
                "teff1": pre["Teff(1)"],
                "teff2": pre["Teff(2)"],
            }

    def test_collapses_follow_the_compas_supernovae(self, converted):
        episodes = [e for e in converted.episodes if e["kind"] == "SN" and e["runs"]]
        events = [e for e in converted.header["events"] if e["kind"] == "collapse"]
        assert len(events) == len(episodes)
        for event, episode in zip(events, episodes, strict=True):
            jumps = [run for run in episode["runs"] if run[2] == "sn_jump"]
            assert event["star"] == episode["star"]
            assert event["during"] == [
                min(r[0] for r in jumps),
                max(r[1] for r in jumps),
            ]
            assert event["until"] == max(r[1] for r in episode["runs"])
            for key, row in (("start", "collapse"), ("end", "post")):
                values = episode["endpoints"][row]
                assert event[key] == {
                    "time": values["Time"],
                    "m1": values["Mass(1)"],
                    "m2": values["Mass(2)"],
                    "a": values["SemiMajorAxis"],
                    "e": values["Eccentricity"],
                }
            progenitor = episode["endpoints"]["collapse"]
            k = episode["star"]
            helium = progenitor[f"Mass_He_Core({k})"]
            carbon = progenitor[f"Mass_CO_Core({k})"]
            assert event["hydrogen"] == max(progenitor[f"Mass({k})"] - helium, 0.0)
            assert event["helium"] == max(helium - carbon, 0.0)

    def test_envelopes_follow_the_compas_common_envelopes(self, converted):
        episodes = [e for e in converted.episodes if e["kind"] == "CE" and e["runs"]]
        events = [e for e in converted.header["events"] if e["kind"] == "envelope"]
        assert len(events) == len(episodes)
        for event, episode in zip(events, episodes, strict=True):
            jumps = [run for run in episode["runs"] if run[2] == "ce_jump"]
            assert event["donor"] == episode["star"]
            assert event["during"] == [
                min(r[0] for r in jumps),
                max(r[1] for r in jumps),
            ]
            assert event["until"] == max(r[1] for r in episode["runs"])
            for key, row in (("start", "pre"), ("end", "post")):
                values = episode["endpoints"][row]
                assert event[key] == {
                    "time": values["Time"],
                    "m1": values["Mass(1)"],
                    "m2": values["Mass(2)"],
                    "a": values["SemiMajorAxis"],
                    "e": values["Eccentricity"],
                    "r1": values["Radius(1)"],
                    "r2": values["Radius(2)"],
                    "teff1": values["Teff(1)"],
                    "teff2": values["Teff(2)"],
                }

    def test_stage_covers_the_episodes_and_labels(self, converted):
        stage = converted.columns["stage"]
        for frame, name in zip(converted.frames, stage, strict=True):
            if frame["eventString"] == "Mass Transfer":
                assert name in ("transfer", "envelope")
        for episode in converted.episodes:
            allowed = {"MT": {"transfer", "envelope"}, "CE": {"envelope"}}
            for first, last, _ in episode["runs"] if episode["kind"] in allowed else ():
                assert set(stage[first : last + 1]) <= allowed[episode["kind"]]


class TestLook:
    def test_palette_is_the_vimes_colour_table(self, converted):
        palette = converted.header["palette"]
        k = 100
        t = palette["temperature"][k]
        assert palette["rgb"][k] == [int(c) for c in temperature_to_rgb(t)]
        assert palette["luminance"][k] == pytest.approx(float(blackbody_color(t)[1]))

    def test_type_icons_and_background(self, converted):
        types = converted.header["types"]
        assert set(types) == set(view3d.TYPE_NAMES)
        assert all(types[name]["icon"] in converted.assets for name in types)
        assert converted.header["sky"] == view3d.SKY
        assert converted.assets[view3d.SKY][:8] == b"\x89PNG\r\n\x1a\n"

    def test_tulips_images_use_the_procedural_sky(self, augmented_copy):
        header, _, assets = strix.read(view3d.convert(augmented_copy, images="tulips"))
        assert header["sky"] is None
        assert view3d.SKY not in assets

    def test_source(self, converted):
        source = converted.header["source"]
        assert source["frames"] == converted.npz.name
        assert (
            source["sha256"] == hashlib.sha256(converted.npz.read_bytes()).hexdigest()
        )


class TestCommand:
    def test_missing_frames_file(self, monkeypatch, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_main(monkeypatch, tmp_path / "missing.npz", "--no-display")

    def test_no_display_only_converts(self, monkeypatch, augmented_copy):
        monkeypatch.setattr(strix, "render", forbidden)
        monkeypatch.setattr(strix, "view", forbidden)
        run_main(monkeypatch, augmented_copy, "--no-display")
        assert augmented_copy.with_suffix(".strix").is_file()

    def test_save_mp4_renders_then_shows(self, monkeypatch, augmented_copy):
        calls = []
        monkeypatch.setattr(
            strix, "render", lambda *a, **k: calls.append(("render", a, k))
        )
        monkeypatch.setattr(strix, "view", lambda *a, **k: calls.append(("view", a, k)))
        run_main(
            monkeypatch, augmented_copy, "--save-mp4", "out.mp4", "--scaling", "log"
        )
        scene = augmented_copy.with_suffix(".strix")
        assert calls == [
            ("render", (scene, "out.mp4"), {"scale": "log", "camera": "fixed"}),
            ("view", (scene,), {"scale": "log", "camera": "fixed"}),
        ]

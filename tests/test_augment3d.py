import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import h5py as h5
import numpy as np
import pytest

from vimes.augment3d import (
    CODE_COLUMNS,
    CONTINUOUS_COLUMNS,
    add_physics_columns,
    check_replay,
    continuous_value,
    load_type4_columns,
    main,
    replay_frames,
    sha256_of,
)
from vimes.preprocess import get_stellar_types, preprocess_to_frames
from vimes.temp_to_color import add_temperatures_and_rgb, interp_from_hdf5

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
FILES = [0, 1, 2]
JUMP_KINDS = ("jump", "phase_jump")
KINDS = ("sample", "jump", "phase_jump", "mt_pad", "ce_pad")


def example(number):
    return EXAMPLES_DIR / f"BSE_Detailed_Output_{number}.h5"


def load_frames(path):
    return np.load(path, allow_pickle=True)["frames"].tolist()


def bit_identical(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        same_layout = a.dtype == b.dtype and a.shape == b.shape
        return same_layout and a.tobytes() == b.tobytes()
    if isinstance(a, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def events_of(made):
    return made["augmented"][0]["x3d.meta"]["events"]


@pytest.fixture(scope="session")
def build(tmp_path_factory):
    made = {}

    def get(number):
        if number not in made:
            folder = tmp_path_factory.mktemp(f"example{number}")
            npz = folder / "frames_data.npz"
            out = folder / "frames_data_3d.npz"
            preprocess_to_frames(example(number), npz)
            add_temperatures_and_rgb(example(number), npz, npz)
            add_physics_columns(example(number), npz, out)
            made[number] = {
                "h5": example(number),
                "npz": npz,
                "out": out,
                "frames": load_frames(npz),
                "augmented": load_frames(out),
            }
        return made[number]

    return get


class TestVimesKeysUntouched:
    @pytest.mark.parametrize("number", FILES)
    def test_original_keys_are_bit_identical(self, build, number):
        made = build(number)
        frames, augmented = made["frames"], made["augmented"]
        assert len(augmented) == len(frames)
        for frame, new in zip(frames, augmented, strict=True):
            assert not any(key.startswith("x3d.") for key in frame)
            for key, value in frame.items():
                assert bit_identical(value, new[key])
            assert all(key.startswith("x3d.") for key in new.keys() - frame.keys())
            assert new["x3d.kind"] in KINDS
            assert [type(row) for row in new["x3d.src"]] == [int, int]

    def test_default_run_leaves_input_alone(self, build, tmp_path, monkeypatch):
        made = build(2)
        frames = tmp_path / "frames_data.npz"
        shutil.copy(made["npz"], frames)
        before = hashlib.sha256(frames.read_bytes()).hexdigest()
        monkeypatch.setattr(sys, "argv", ["augment3d", str(made["h5"]), str(frames)])
        main()
        assert hashlib.sha256(frames.read_bytes()).hexdigest() == before
        assert sha256_of(frames) == before
        assert (tmp_path / "frames_data_3d.npz").exists()

    def test_in_place_with_output_is_an_error(self, monkeypatch):
        argv = ["augment3d", str(example(0)), "f.npz", "o.npz", "--in-place"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2


class TestReplay:
    @pytest.mark.parametrize("number", FILES)
    def test_replay_equals_npz(self, build, number):
        frames = build(number)["frames"]
        replayed, sources = replay_frames(example(number))
        check_replay(frames, replayed)
        assert len(sources) == len(frames)
        with pytest.raises(ValueError, match=f"^frame {len(frames) - 1}: "):
            check_replay(frames[:-1], replayed)

    def test_perturbed_time_is_rejected(self, build, tmp_path):
        frames = load_frames(build(0)["npz"])
        k = len(frames) // 2
        frames[k]["Time"] = float(np.nextafter(frames[k]["Time"], np.inf))
        perturbed = tmp_path / "perturbed.npz"
        np.savez_compressed(perturbed, frames=frames)
        with pytest.raises(ValueError, match=f"^frame {k}: "):
            add_physics_columns(example(0), perturbed, tmp_path / "out.npz")


class TestInterpolationParity:
    @pytest.mark.parametrize("number", FILES)
    def test_teff_through_column_path(self, build, number):
        frames = build(number)["frames"]
        time, columns = load_type4_columns(example(number), ["Teff(1)", "Teff(2)"])
        for frame in frames:
            t = frame["Time"]
            assert continuous_value(t, time, columns["Teff(1)"]) == frame["Teff1"]
            assert continuous_value(t, time, columns["Teff(2)"]) == frame["Teff2"]

    @pytest.mark.parametrize("number", FILES)
    def test_columns_match_independent_mapping(self, build, number):
        names = CONTINUOUS_COLUMNS + CODE_COLUMNS
        with h5.File(example(number), "r") as handle:
            mask = handle["Record_Type"][()] == 4
            time = handle["Time"][()][mask]
            values = {name: handle[name][()][mask] for name in names}
        for frame in build(number)["augmented"]:
            t = frame["Time"]
            for name in CONTINUOUS_COLUMNS:
                expected = float(interp_from_hdf5(t, time, values[name]))
                assert type(frame["x3d." + name]) is float
                assert frame["x3d." + name] == expected
            earlier = np.flatnonzero(time <= t)
            held = earlier[-1] if len(earlier) else 0
            for name in CODE_COLUMNS:
                assert type(frame["x3d." + name]) is int
                assert frame["x3d." + name] == int(values[name][held])


class TestUnits:
    def test_wrong_units_are_rejected(self, build, tmp_path):
        made = build(2)
        broken = tmp_path / "broken.h5"
        shutil.copy(made["h5"], broken)
        with h5.File(broken, "r+") as handle:
            handle["RocheLobe(1)"].attrs["units"] = np.bytes_(b"km")
        with pytest.raises(ValueError, match=re.escape("RocheLobe(1)")):
            add_physics_columns(broken, made["npz"], tmp_path / "out.npz")

    @pytest.mark.parametrize("number", FILES)
    def test_meta_units_match_file(self, build, number):
        units = build(number)["augmented"][0]["x3d.meta"]["units"]
        with h5.File(example(number), "r") as handle:
            for name, expected in units.items():
                assert handle[name].attrs["units"].decode() == expected


class TestEvents:
    @classmethod
    def setup_class(cls):
        cls.expected = {
            0: [("MT", 721), ("MT", 802), ("SN", 2680), ("MT", 3040), ("SN", 4588)],
            1: [("MT", 2003), ("SN", 2159), ("CE", 3809), ("SN", 3947)],
            2: [("MT", 694), ("SN", 1042), ("SN", 4563)],
        }
        cls.collapse = {0: [2677, 4585], 1: [2156, 3944], 2: [1039, 4560]}
        cls.key_role = {"MT": "start", "CE": "close", "SN": "post"}

    @pytest.mark.parametrize("number", FILES)
    def test_event_list(self, build, number):
        events = events_of(build(number))
        found = [(e["kind"], e["rows"][self.key_role[e["kind"]]]) for e in events]
        assert found == self.expected[number]
        assert [e["id"] for e in events] == list(range(len(events)))
        supernovae = [e for e in events if e["kind"] == "SN"]
        assert [e["star"] for e in supernovae] == [1, 2]
        assert [e["rows"]["collapse"] for e in supernovae] == self.collapse[number]
        assert all(e["rows"]["preSN"] == e["rows"]["post"] - 1 for e in supernovae)
        for event in events:
            for values in event["endpoints"].values():
                assert all(type(value) in (int, float) for value in values.values())
        json.dumps(build(number)["augmented"][0]["x3d.meta"])

    def test_common_envelope_rows(self, build):
        (ce,) = [e for e in events_of(build(1)) if e["kind"] == "CE"]
        expected = {
            "pre": 3802,
            "typeChange": 3805,
            "post": 3806,
            "close": 3809,
            "next": 3815,
        }
        assert ce["rows"] == expected
        assert ce["star"] == 2
        assert set(ce["endpoints"]) == {"pre", "typeChange", "post"}

    def test_mass_transfer_rows(self, build):
        mts = [e for e in events_of(build(0)) if e["kind"] == "MT"]
        found = [(e["rows"]["pre"], e["rows"]["post"]) for e in mts]
        assert found == [(715, 727), (794, 802), (3032, 3040)]
        assert [e["rows"]["start"] for e in mts] == [721, 802, 3040]
        assert [e["star"] for e in mts] == [1, 1, 2]


class TestAttachment:
    def test_common_envelope_frames(self, build):
        frames = build(1)["augmented"]
        (ce,) = [e for e in events_of(build(1)) if e["kind"] == "CE"]
        expected_runs = [
            [1110, 1159, "ce_jump"],
            [1161, 1220, "ce"],
            [1246, 1305, "ce"],
        ]
        assert ce["runs"] == expected_runs
        for i in range(1110, 1160):
            assert frames[i]["x3d.kind"] == "phase_jump"
            assert frames[i]["x3d.src"] == [3802, 3808]
        labelled = [f for f in frames if f["eventString"] == "Common Envelope"]
        assert len(labelled) == 120
        for frame in labelled:
            assert frame["x3d.ep"] == ce["id"]
            assert frame["x3d.epPre"]["row"] == 3802
            assert frame["x3d.epTypeChange"]["row"] == 3805
            assert frame["x3d.epPost"]["row"] == 3806

    @pytest.mark.parametrize("number", FILES)
    def test_supernova_runs(self, build, number):
        frames = build(number)["augmented"]
        type_map = get_stellar_types()
        for event in events_of(build(number)):
            if event["kind"] != "SN":
                continue
            explosion = event["rows"]["post"]
            jumps = [run for run in event["runs"] if run[2] == "sn_jump"]
            windows = [run for run in event["runs"] if run[2] == "sn_window"]
            assert len(jumps) == 1
            assert len(windows) == 1
            first, last, _ = jumps[0]
            assert last - first + 1 == 50
            assert frames[first - 1]["x3d.kind"] not in JUMP_KINDS
            assert frames[last + 1]["x3d.kind"] not in JUMP_KINDS
            for i in range(first, last + 1):
                low, high = frames[i]["x3d.src"]
                assert frames[i]["x3d.kind"] in JUMP_KINDS
                assert low < explosion < high
            assert windows[0][:2] == [last + 1, last + 50]
            star = event["star"]
            new_type = type_map(event["endpoints"]["post"][f"Stellar_Type({star})"])
            assert new_type in ("BH", "NS")
            assert frames[last + 1][f"stypeName{star}"] == new_type

    @pytest.mark.parametrize("number", FILES)
    def test_mass_transfer_runs(self, build, number):
        frames = build(number)["augmented"]
        for event in events_of(build(number)):
            if event["kind"] != "MT":
                continue
            pre, start = event["rows"]["pre"], event["rows"]["start"]
            parts = [part for _, _, part in event["runs"]]
            assert parts.count("mt") == 1 and set(parts) == {"mt", "mt_jump"}
            for first, last, part in event["runs"]:
                for i in range(first, last + 1):
                    if part == "mt":
                        assert frames[i]["eventString"] == "Mass Transfer"
                    else:
                        low, high = frames[i]["x3d.src"]
                        assert frames[i]["x3d.kind"] in JUMP_KINDS
                        assert low <= pre and start <= high
                if part == "mt":
                    assert last - first + 1 == 60
        for frame in frames:
            if frame["eventString"] == "Mass Transfer":
                assert frame["x3d.epKind"] == "mt"

    @pytest.mark.parametrize("number", FILES)
    def test_episode_keys_match_runs(self, build, number):
        frames = build(number)["augmented"]
        meta = frames[0]["x3d.meta"]
        expected = [(-1, "", -1, 0)] * len(frames)
        for event in meta["events"]:
            assert bool(event["runs"]) != (event["id"] in meta["unattached"])
            for first, last, part in event["runs"]:
                for k, i in enumerate(range(first, last + 1)):
                    assert expected[i][0] == -1
                    expected[i] = (event["id"], part, k, last - first + 1)
        keys = ("x3d.ep", "x3d.epKind", "x3d.epIndex", "x3d.epCount")
        for frame, episode in zip(frames, expected, strict=True):
            assert tuple(frame[key] for key in keys) == episode
            assert ("x3d.epPre" in frame) == (episode[0] != -1)
            assert ("x3d.epPost" in frame) == (episode[0] != -1)


@pytest.mark.skipif(os.environ.get("VIMES_SLOW") != "1", reason="set VIMES_SLOW=1")
class TestPygameUnaffected:
    @pytest.mark.parametrize("number", FILES)
    def test_same_decoded_frames(self, build, number, tmp_path):
        made = build(number)
        vimes = Path(sys.executable).with_name("vimes")
        env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
        checksums = []
        for frames in (made["npz"], made["out"]):
            video = tmp_path / f"{frames.stem}.mp4"
            cmd = [str(vimes), str(frames), "--no-display", "--save-mp4", str(video)]
            subprocess.run(cmd, check=True, env=env, cwd=tmp_path)
            md5 = ["ffmpeg", "-v", "error", "-i", str(video), "-f", "framemd5", "-"]
            result = subprocess.run(md5, check=True, capture_output=True)
            checksums.append(result.stdout)
        assert checksums[0] == checksums[1]

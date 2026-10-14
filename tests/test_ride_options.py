import json

import pytest
from pydantic import ValidationError

from groovemap import library, pipeline
from groovemap.server import OptionsPatch


@pytest.mark.parametrize("value", [True, False])
def test_ride_options_round_trip(tmp_path, value):
    patch = OptionsPatch(ride_runs=value)
    options = pipeline.Options(**patch.model_dump())
    library.save_options(tmp_path, options)
    assert library.load_options(tmp_path)["ride_runs"] is value
    assert json.loads((tmp_path / "options.json").read_text())["ride_runs"] is value
    library.save_options(tmp_path, library.DEFAULT_OPTIONS | {"ride_runs": value})
    assert library.load_options(tmp_path)["ride_runs"] is value


@pytest.mark.parametrize("value", [0, 1, "true", "false", None, [], {}])
def test_ride_options_reject_non_boolean(value):
    with pytest.raises(ValidationError):
        OptionsPatch(ride_runs=value)


def test_ride_options_default_and_legacy(tmp_path):
    assert pipeline.Options().ride_runs is True
    assert OptionsPatch().ride_runs is True
    assert library.load_options(tmp_path)["ride_runs"] is True
    (tmp_path / "options.json").write_text('{"grid": 8, "sound_names": {"cymbal-1": "ride_bell"}}')
    options = library.load_options(tmp_path)
    assert options["ride_runs"] is True
    assert options["grid"] == 8
    assert options["sound_names"] == {"cymbal-1": "ride_bell"}

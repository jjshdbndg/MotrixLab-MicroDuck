from pathlib import Path

import motrixsim as mtx

MODEL_FILE = (
    Path(__file__).parents[1]
    / "motrix_envs"
    / "src"
    / "motrix_envs"
    / "locomotion"
    / "microduck"
    / "xmls"
    / "scene_rollers.xml"
)


def test_microduck_passive_roller_model_contract():
    model = mtx.load_model(str(MODEL_FILE))

    assert model.num_actuators == 14
    passive_wheels = [name for name in model.joint_names if name.startswith("passive_") and "wheel" in name]
    assert len(passive_wheels) >= 4
    assert not any(name.startswith("passive_") for name in model.actuator_names)
    assert set(passive_wheels) >= {
        "passive_LF_wheel",
        "passive_LR_wheel",
        "passive_RF_wheel",
        "passive_RR_wheel",
    }

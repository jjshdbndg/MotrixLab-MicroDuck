from pathlib import Path

import motrixsim as mtx

MODEL_FILE = (
    Path(__file__).parents[1]
    / "motrix_envs"
    / "src"
    / "motrix_envs"
    / "basic"
    / "point_mass"
    / "point_mass.xml"
)


def test_point_mass_model_loads_with_joint_controlled_target():
    model = mtx.load_model(str(MODEL_FILE))

    assert model.num_actuators == 2

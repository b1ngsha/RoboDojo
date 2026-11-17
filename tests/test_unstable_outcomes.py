"""Exercise the real eval entry points with CPU substitutes for Isaac and policy I/O."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cpu_eval(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBODOJO_RUN_ID", "cpu-run")
    monkeypatch.delenv("EVAL_NUM", raising=False)
    scenario = {}

    def install_module(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class UnStableError(Exception):
        pass

    install_module("utils.cluttered_generator", UnStableError=UnStableError)
    install_module("transforms3d")

    class ObsManager:
        def __init__(self, **kwargs):
            pass

        def reset(self):
            pass

        def get_obs(self):
            return {}

    install_module("env.observation_manager.obs_manager", ObsManager=ObsManager)

    class WsModelClient:
        def __init__(self, **kwargs):
            pass

        def call(self, **kwargs):
            pass

        def close(self):
            pass

    install_module("client_server.ws.model_client", WsModelClient=WsModelClient)

    class CpuTask:
        def __init__(self, config, app, **kwargs):
            self.num_envs = config.eval_cfg.num_envs
            self.dt = 0.01
            self.env_seed_list = [0] * self.num_envs
            self.scene_manager = SimpleNamespace(
                layout_manager=SimpleNamespace(
                    set_saved_layout=lambda *args: None,
                    layout_valid=[True] * self.num_envs,
                    check_layout_stability=self.check_layout_stability,
                ),
                apply_saved_poses=lambda **kwargs: None,
            )
            self.robot_manager = SimpleNamespace(
                set_origin_endpose=lambda: None,
                set_robot_init_state=lambda: None,
            )
            self.reward_manager = SimpleNamespace(
                init_state=lambda: None,
                get_score=lambda: [25] * self.num_envs,
            )

        def check_layout_stability(self, env):
            unstable = [i for i, seed in enumerate(env.env_seeds) if scenario.get(seed) == "reset_unstable"]
            return not unstable, unstable

        def reset(self, **kwargs):
            outcome = scenario.get(self.env_seeds[0])
            if isinstance(outcome, Exception):
                raise outcome

        def run_reward(self):
            pass

        def get_score(self):
            pass

        def render(self):
            pass

        def sim_step(self):
            pass

        def close(self):
            pass

    registry = importlib.import_module("task.RoboDojo.task_registry")
    monkeypatch.setattr(registry, "load_task_class", lambda name: (name, CpuTask))

    # Real SeedManager assigns ordinals 0..3 to these noncontiguous public suffixes.
    seeds = importlib.import_module("env.seed_manager.seed_manager")
    assets = tmp_path / "assets"
    layouts = assets / "Eval_Layout/RoboDojo/arx_x5/0"
    layouts.mkdir(parents=True)
    for suffix in (7, 42, 100, 900):
        (layouts / f"stack_bowls_{suffix}.json").write_text("{}")
    monkeypatch.setattr(seeds, "ASSETS_PATH", str(assets))

    def eval_one_episode(TASK_ENV, model_client):
        env = TASK_ENV
        outcome = scenario.get(env.env_seeds[0], "success")
        if outcome == "evaluation_exception":
            raise UnStableError("policy raised during evaluation")
        if outcome == "evaluation_error":
            raise RuntimeError("policy failed; not an unstable layout")
        env.success = [outcome == "success"]
        if outcome == "evaluation_unstable":
            env.mark_env_unstable(0)
        # CPU substitute for the ffmpeg process; real save_video/abort routing runs.
        stream = Path(env._stream_dir) / "env0_camera.tmp.mp4"
        stream.parent.mkdir(parents=True, exist_ok=True)
        stream.write_bytes(b"encoded frames")
        env.video_writers[0] = {
            "camera": SimpleNamespace(
                out_path=str(stream),
                close=lambda **kwargs: None,
                abort=lambda: stream.unlink(missing_ok=True),
                n_frames=1,
                width=1,
                height=1,
                fps=30,
            )
        }

    install_module("XPolicyLab.policy.cpu.deploy", eval_one_episode=eval_one_episode)

    def load_source_module(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    eval_module = load_source_module("src.eval_client.eval_env", "src/eval_client/eval_env.py")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            pass

        def __init__(self, args):
            self.app = SimpleNamespace(close=lambda: None)

    install_module("isaaclab.app", AppLauncher=AppLauncher)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--task_name",
            "stack_bowls",
            "--env_cfg_type",
            "arx_x5",
            "--device_id",
            "0",
            "--policy_name",
            "cpu",
            "--port",
            "9999",
            "--additional_info",
            "cpu",
            "--seed",
            "0",
        ],
    )
    main = load_source_module("src.eval_client.main", "src/eval_client/main.py")
    save_dir = tmp_path / "eval_result/RoboDojo/stack_bowls/cpu/arx_x5/0_cpu/cpu-run"

    def make_env():
        from omegaconf import OmegaConf

        return eval_module.create_eval_env(
            OmegaConf.create(
                {
                    "eval_cfg": {
                        "task_name": "stack_bowls",
                        "config_name": "arx_x5",
                        "num_envs": 1,
                        "policy_name": "cpu",
                        "additional_info": "cpu",
                        "seed": 0,
                        "config": {"robot": "dual_x5"},
                    },
                    "deploy_cfg": {"policy_name": "cpu", "port": 9999},
                }
            ),
            None,
        )

    return SimpleNamespace(
        main=main,
        make_env=make_env,
        scenario=scenario,
        save_dir=save_dir,
    )


def test_reset_unstable_records_current_native_indices_cumulatively(cpu_eval):
    cpu_eval.scenario.update({0: "reset_unstable", 1: "reset_unstable"})

    cpu_eval.main.main()

    assert json.loads((cpu_eval.save_dir / "_skips.json").read_text()) == {
        "details": {
            "0": {"reason": "unstable", "phase": "reset"},
            "1": {"reason": "unstable", "phase": "reset"},
        }
    }
    result = json.loads((cpu_eval.save_dir / "_result.json").read_text())
    assert result["eval_time"] == 2
    assert [detail["layout_id"] for detail in result["details"].values()] == [2, 3]
    assert len(list(cpu_eval.save_dir.glob("*.mp4"))) == 2


def test_run_eval_records_only_unstable_layouts_and_preserves_valid_outcomes(cpu_eval):
    cpu_eval.scenario.update({0: "evaluation_unstable", 1: "fail", 2: "evaluation_unstable"})
    env = cpu_eval.make_env()

    for seed in (0, 1, 2, 3):
        env.reset(seed=[seed])
        # An unrelated marker must not create a receipt for an unevaluated env.
        env.mark_env_unstable(99)
        env.run_eval()

    assert json.loads((cpu_eval.save_dir / "_skips.json").read_text()) == {
        "details": {
            "0": {"reason": "unstable", "phase": "evaluation"},
            "2": {"reason": "unstable", "phase": "evaluation"},
        }
    }
    assert (env.success_nums, env.fail_nums, env.unstable_nums) == (1, 1, 2)
    assert json.loads((cpu_eval.save_dir / "_result.json").read_text()) == {
        "success_rate": 0.5,
        "eval_time": 2,
        "score": 62.5,
        "details": {
            "0": {"layout_id": 1, "success": False, "score": 0.25},
            "1": {"layout_id": 3, "success": True, "score": 1.0},
        },
    }
    assert sorted(path.name for path in cpu_eval.save_dir.glob("*.mp4")) == [
        "episode_0000000_camera_fail.mp4",
        "episode_0000001_camera_success.mp4",
    ]
    assert list((cpu_eval.save_dir / "_stream").iterdir()) == []


@pytest.mark.parametrize("failure", ["reset", "evaluation", "physx_broken", "physx_fatal", "unstable_evaluation"])
def test_exceptions_do_not_fabricate_unstable_receipts(cpu_eval, failure, capsys):
    if failure == "reset":
        cpu_eval.scenario[0] = RuntimeError("UnStableError text is not an unstable exception")
    elif failure == "evaluation":
        cpu_eval.scenario[0] = "evaluation_error"
    elif failure == "unstable_evaluation":
        cpu_eval.scenario[0] = "evaluation_exception"
    elif failure == "physx_broken":
        error = cpu_eval.main.PhysXBrokenError("invalid PhysX transform")
        error.broken_envs = {0}
        cpu_eval.scenario[0] = error
    else:
        cpu_eval.scenario[0] = cpu_eval.main.PhysXFatalError("CUDA context lost")

    if failure == "physx_fatal":
        with pytest.raises(cpu_eval.main.PhysXFatalError, match="CUDA context lost"):
            cpu_eval.main.main()
        assert not (cpu_eval.save_dir / "_result.json").exists()
    else:
        cpu_eval.main.main()
        result = json.loads((cpu_eval.save_dir / "_result.json").read_text())
        assert [detail["layout_id"] for detail in result["details"].values()] == [1, 2, 3]
        assert result["eval_time"] == 3
        if failure in ("reset", "evaluation"):
            assert "unhandled exception during reset/run_eval: RuntimeError" in capsys.readouterr().out
    assert not (cpu_eval.save_dir / "_skips.json").exists()


def test_reset_and_evaluation_receipts_share_one_run_sidecar(cpu_eval):
    cpu_eval.scenario.update({0: "reset_unstable", 1: "evaluation_unstable"})

    cpu_eval.main.main()

    assert json.loads((cpu_eval.save_dir / "_skips.json").read_text()) == {
        "details": {
            "0": {"reason": "unstable", "phase": "reset"},
            "1": {"reason": "unstable", "phase": "evaluation"},
        }
    }


def test_failed_atomic_publish_preserves_previous_receipts(cpu_eval, monkeypatch):
    cpu_eval.scenario[0] = "evaluation_unstable"
    env = cpu_eval.make_env()
    env.reset(seed=[0])
    env.run_eval()
    path = cpu_eval.save_dir / "_skips.json"
    previous = path.read_bytes()

    from utils import save_file

    original_replace = save_file.os.replace

    def fail_skip_publish(source, destination):
        if Path(destination) == Path(env.save_dir) / "_skips.json":
            raise OSError("publish failed")
        original_replace(source, destination)

    monkeypatch.setattr(save_file.os, "replace", fail_skip_publish)
    cpu_eval.scenario[1] = "evaluation_unstable"
    env.reset(seed=[1])
    with pytest.raises(OSError, match="publish failed"):
        env.run_eval()

    assert path.read_bytes() == previous
    assert not path.with_suffix(".json.tmp").exists()

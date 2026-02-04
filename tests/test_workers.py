import numpy as np
import pytest

from app.ui.workers import OptimizerWorker, DisplacementWorker


def test_optimizer_worker(monkeypatch):
    recorded = {"progress": [], "frame": None, "finished": None, "error": None}

    def dummy_optimize(**kwargs):
        progress_cb = kwargs.get("progress_callback")
        # call progress callback once with a dummy frame
        progress_cb(1, 3.14, 0.01, np.array([[42]]))
        return np.array([0.5]), np.array([0.1])

    monkeypatch.setattr("app.ui.workers.optimizers.optimize", dummy_optimize)

    worker = OptimizerWorker({"some_param": 1})

    worker.progress.connect(lambda it, obj, ch: recorded["progress"].append((it, obj, ch)))
    worker.frameReady.connect(lambda frame: recorded.update({"frame": frame}))
    worker.finished.connect(lambda res: recorded.update({"finished": res}))
    worker.error.connect(lambda e: recorded.update({"error": e}))

    # Run synchronously in the test thread
    worker.run()

    assert recorded["progress"] == [(1, 3.14, 0.01)]
    assert isinstance(recorded["frame"], np.ndarray)
    assert recorded["frame"].shape == (1, 1) or recorded["frame"].size == 1
    np.testing.assert_array_equal(recorded["finished"][0], np.array([0.5]))
    np.testing.assert_array_equal(recorded["finished"][1], np.array([0.1]))
    assert isinstance(recorded["finished"], tuple)
    assert recorded["error"] is None


def test_displacement_worker(monkeypatch):
    frames = [np.array([1]), np.array([2])]

    def dummy_generator(params, xPhys, progress_cb):
        for i, f in enumerate(frames):
            stop = progress_cb(i)
            yield f
            if stop:
                break

    monkeypatch.setattr(
        "app.ui.workers.displacements.run_iterative_displacement", dummy_generator
    )

    recorded = {"progress": [], "frames": [], "finished": None, "error": None}

    worker = DisplacementWorker({"p": 1}, np.array([0.0]), np.array([0.0]))

    worker.progress.connect(lambda it: recorded["progress"].append(it))
    worker.frameReady.connect(lambda f: recorded["frames"].append(np.array(f)))
    worker.finished.connect(lambda msg, flag: recorded.update({"finished": (msg, flag)}))
    worker.error.connect(lambda e: recorded.update({"error": e}))

    # Run synchronously in the test thread
    worker.run()

    assert recorded["progress"] == [0, 1]
    assert len(recorded["frames"]) == 2
    np.testing.assert_array_equal(recorded["frames"][0], frames[0])
    np.testing.assert_array_equal(recorded["frames"][1], frames[1])
    assert recorded["finished"] is not None
    assert recorded["error"] is None

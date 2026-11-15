"""Check the actual author loss, its gradients, and optimizer restoration."""
import ast
from types import SimpleNamespace

import numpy as np
import pytest

from fault_workflow.baseline import (AUTHOR_DIR, compile_author_model, configure_tensorflow,
                                     load_training_model, require_checkpoint_recipe_match, training_settings)
from fault_workflow.common import WORKSPACE, load_config


@pytest.fixture(scope="module")
def runtime():
    config = load_config(WORKSPACE / "configs/baseline_balanced.json")
    return configure_tensorflow(config, "cpu"), config


def author_loss_function(tf, config):
    """Execute only pinned author loss definitions; adapt renamed TensorFlow APIs."""
    path = config["data_root"] / AUTHOR_DIR / "unet3.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [n for n in module.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("cross_entropy_balanced", "_to_tensor")]

    class OldTensorFlow:
        log = staticmethod(tf.math.log)
        nn = SimpleNamespace(weighted_cross_entropy_with_logits=lambda logits, targets, pos_weight:
                             tf.nn.weighted_cross_entropy_with_logits(labels=targets, logits=logits, pos_weight=pos_weight))

        def __getattr__(self, name):
            return getattr(tf, name)

    namespace = {"tf": OldTensorFlow(), "K": tf.keras.backend}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["cross_entropy_balanced"]


def test_checkpoint_recipe_has_explicit_independent_configuration(runtime):
    _, config = runtime
    assert training_settings(config)["loss"] == "cross_entropy_balanced"
    assert config["output_root"].name == "server_balanced_v2"
    require_checkpoint_recipe_match(config, training_settings(config))


def test_balanced_loss_and_gradients_match_pinned_author_formula(runtime):
    tf, config = runtime
    from fault_workflow.losses import cross_entropy_balanced

    original = author_loss_function(tf, config)
    labels = tf.constant([0, 0, 1, 0, 0, 1, 0, 0], dtype=tf.float32)
    probability = tf.Variable([0.03, 0.16, 0.29, 0.42, 0.55, 0.68, 0.81, 0.97], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        expected = original(labels, probability)
        actual = cross_entropy_balanced(labels, probability)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(tape.gradient(actual, probability).numpy(), tape.gradient(expected, probability).numpy(),
                               rtol=2e-6, atol=1e-7)
    # A confident sparse positive must matter more than each single negative.
    np.testing.assert_allclose(actual.numpy(), -np.mean(0.75 * labels.numpy() * np.log(probability.numpy())
                               + 0.25 * (1 - labels.numpy()) * np.log1p(-probability.numpy())), rtol=2e-6)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_empty_and_full_targets_have_finite_zero_loss_and_gradient(runtime, label):
    tf, _ = runtime
    from fault_workflow.losses import cross_entropy_balanced

    probability = tf.Variable([0, 0.3, 0.9, 1], dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = cross_entropy_balanced(tf.fill([4], label), probability)
    assert float(loss) == 0
    np.testing.assert_array_equal(tape.gradient(loss, probability).numpy(), np.zeros(4))


def test_balanced_keras_serialization_restores_identical_next_update(runtime, tmp_path):
    tf, config = runtime
    model = tf.keras.Sequential([tf.keras.layers.Input((3,)), tf.keras.layers.Dense(1, activation="sigmoid")])
    compile_author_model(tf, model, training_settings(config))
    x = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    y = np.array([[0], [1], [0], [0]], dtype=np.float32)
    model.train_on_batch(x, y)
    path = tmp_path / "balanced.keras"
    model.save(path)
    restored = load_training_model(tf, path)
    assert int(restored.optimizer.iterations) == 1
    np.testing.assert_array_equal(model(x).numpy(), restored(x).numpy())
    model.reset_metrics()
    restored.reset_metrics()
    model.train_on_batch(x, y)
    restored.train_on_batch(x, y)
    for expected, actual in zip(model.get_weights(), restored.get_weights(), strict=True):
        np.testing.assert_array_equal(expected, actual)

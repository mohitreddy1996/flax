# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for flax.examples.sst2.train."""

from absl.testing import absltest

import jax

import tensorflow_datasets as tfds

import flax

import input_pipeline
import model as sst2_model
import train


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class TrainTest(absltest.TestCase):

  def test_single_train_step(self):
    batch_size = 32
    seed = 0
    data_source = input_pipeline.SST2DataSource(min_freq=5)
    model = sst2_model.create_model(
        seed,
        batch_size,
        20,
        dict(
            vocab_size=data_source.vocab_size,
            embedding_size=256,
            hidden_size=256,
            output_size=1,
            unk_idx=data_source.unk_idx,
            dropout=0.5,
            emb_dropout=0.5,
            word_dropout_rate=0.1))
    optimizer = flax.optim.Adam(learning_rate=0.0005).create(model)

    train_batches = input_pipeline.get_batches(
        dataset=data_source.train_dataset, batch_size=batch_size)
    ex = next(iter(tfds.as_numpy(train_batches)))

    _, loss, _ = train.train_step(
        optimizer=optimizer, inputs=ex['sentence'], lengths=ex['length'],
        labels=ex['label'], rng=jax.random.PRNGKey(seed=seed), l2_reg=1e-6)

    self.assertGreaterEqual(loss, 0.0)


if __name__ == '__main__':
  absltest.main()

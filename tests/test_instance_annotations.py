"""CPU-only tests: python -m unittest discover -s tests -p test_instance_annotations.py"""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from data_processing.instance_annotations import InstanceAnnotations, transform_pair


class InstanceAnnotationTests(unittest.TestCase):
    def fixture(self):
        raw = np.array([[[0, 999, 999, 0], [0, 999, 70000, 0]],
                        [[0, 0, 70000, 0], [999, 999, 0, 0]]], dtype=np.uint32)
        ann = InstanceAnnotations.from_renderer(
            raw, renderer_objects={0: 'background', 999: 'bread', 70000: 'basket'},
            dataset_id='sample', episode_id='ep_0', camera='head', frame_ids=[0, 1])
        palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        return palette[ann.labels], ann

    def transform(self, rgb, ann, **kwargs):
        return transform_pair(rgb, ann, stream_key=ann.stream_key, frame_ids=ann.frame_ids, **kwargs)

    def test_large_renderer_ids_compact_and_parts_merge(self):
        _, ann = self.fixture()
        self.assertEqual(ann.labels.dtype, np.uint8)
        self.assertEqual(ann.objects, ('background', 'basket', 'bread'))
        self.assertEqual(int(ann.mask('bread').sum()), 5)
        merged = InstanceAnnotations.from_renderer(
            np.array([[[1000, 1001]]]), renderer_objects={1000: 'robot', 1001: 'robot'},
            dataset_id='d', episode_id='e', camera='c', frame_ids=[0])
        np.testing.assert_array_equal(merged.labels, [[[1, 1]]])

    def test_unknown_renderer_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            InstanceAnnotations.from_renderer(np.array([[[5]]]), renderer_objects={0: 'background'},
                dataset_id='d', episode_id='e', camera='c', frame_ids=[0])

    def test_more_than_256_objects_upgrade(self):
        ann = InstanceAnnotations.from_renderer(np.arange(300).reshape(1, 10, 30),
            renderer_objects={i: f'object_{i}' for i in range(300)}, dataset_id='d',
            episode_id='e', camera='c', frame_ids=[0])
        self.assertEqual(ann.labels.dtype, np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.h5'
            ann.save(path)
            loaded = InstanceAnnotations.load(path)
            np.testing.assert_array_equal(loaded.labels, ann.labels)
            self.assertEqual(loaded.objects, ann.objects)

    def test_roundtrip_and_compression(self):
        _, ann = self.fixture()
        labels = np.zeros((8, 128, 128), dtype=np.uint8)
        labels[:, 30:90, 25:85] = 2
        ann = replace(ann, labels=labels, frame_ids=np.arange(8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.h5'
            ann.save(path)
            loaded = InstanceAnnotations.load(path)
            np.testing.assert_array_equal(loaded.labels, labels)
            self.assertEqual(loaded.stream_key, ann.stream_key)
            with h5py.File(path) as f:
                self.assertEqual(f['labels'].compression, 'gzip')
                self.assertLess(f['labels'].id.get_storage_size(), labels.nbytes // 10)
                self.assertFalse(any('hash' in name for name in f.attrs))
                self.assertFalse(f['labels'].fletcher32)

    def test_crop_flip_and_frame_selection_stay_aligned(self):
        rgb, ann = self.fixture()
        out, mapped = self.transform(rgb, ann, indices=[1, 0], crop=(1, 0, 4, 2), flip_horizontal=True)
        np.testing.assert_array_equal(out, rgb[[1, 0], :, 1:4][:, :, ::-1])
        np.testing.assert_array_equal(mapped.labels, ann.labels[[1, 0], :, 1:4][:, :, ::-1])
        np.testing.assert_array_equal(mapped.frame_ids, [1, 0])
        self.assertEqual(mapped.history[-1]['crop'], [1, 0, 4, 2])

    def test_resize_nearest_preserves_ids_and_centers(self):
        rgb, ann = self.fixture()
        out, mapped = self.transform(rgb, ann, size=(2, 1))
        self.assertEqual(out.shape, (2, 1, 2, 3))
        np.testing.assert_array_equal(mapped.labels, ann.labels[:, 1:2, [1, 3]])
        self.assertTrue(set(np.unique(mapped.labels)) <= set(np.unique(ann.labels)))
        _, up = self.transform(rgb, ann, size=(8, 4))
        np.testing.assert_array_equal(up.labels, np.repeat(np.repeat(ann.labels, 2, axis=1), 2, axis=2))

    def test_changed_rgb_values_do_not_change_labels(self):
        rgb, ann = self.fixture()
        _, mapped = self.transform(255-rgb, ann)
        np.testing.assert_array_equal(mapped.labels, ann.labels)

    def test_drop_last_observation_and_persist_history(self):
        rgb, ann = self.fixture()
        out, mapped = self.transform(rgb, ann, indices=[0], size=(3, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.h5'
            mapped.save(path)
            loaded = InstanceAnnotations.load(path)
            self.assertEqual(loaded.history, mapped.history)
            self.assertEqual(out.shape[0], len(loaded.frame_ids))
            np.testing.assert_array_equal(loaded.frame_ids, [0])

    def test_wrong_stream_frames_and_shape_rejected(self):
        rgb, ann = self.fixture()
        for stream, ids, images in [('wrong', ann.frame_ids, rgb),
                                     (ann.stream_key, [1, 0], rgb),
                                     (ann.stream_key, [0, 1], rgb[:, :, :2])]:
            with self.subTest(stream=stream, ids=ids), self.assertRaises(ValueError):
                transform_pair(images, ann, stream_key=stream, frame_ids=ids)

    def test_invalid_metadata_and_transforms_rejected(self):
        rgb, ann = self.fixture()
        for bad in [replace(ann, frame_ids=np.array([0, 0])),
                    replace(ann, labels=ann.labels.astype(float)),
                    replace(ann, labels=np.full_like(ann.labels, 9)),
                    replace(ann, objects=('background', 'bread', 'bread'))]:
            with self.assertRaises(ValueError):
                bad.validate()
        for options in [dict(indices=[0, 0]), dict(indices=[-1]), dict(indices=[2]),
                        dict(crop=(0, 0, 5, 2)), dict(size=(0, 2))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.transform(rgb, ann, **options)

    def test_schema_and_missing_fields_rejected(self):
        _, ann = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.h5'
            ann.save(path)
            with h5py.File(path, 'a') as f:
                f.attrs['schema_version'] = 99
            with self.assertRaisesRegex(ValueError, 'schema'):
                InstanceAnnotations.load(path)
            ann.save(path)
            with h5py.File(path, 'a') as f:
                del f['frame_ids']
            with self.assertRaisesRegex(ValueError, 'missing'):
                InstanceAnnotations.load(path)


if __name__ == '__main__':
    unittest.main()

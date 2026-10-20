"""CPU-only instance labels, compressed sidecars and paired image transforms.

No simulator, policy or hashing dependency. Labels describe visible instances,
not mixed RGB boundary colors. Stream IDs are caller-assigned, not content IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from PIL import Image

SCHEMA_VERSION = 1


def _label_dtype(count: int):
    if count <= 256:
        return np.uint8
    if count <= 65536:
        return np.uint16
    return np.uint32


@dataclass(frozen=True)
class InstanceAnnotations:
    """One episode/camera, with original frame IDs retained after subsampling.

    objects[0] is the background; other entries are unique logical object names.
    Several renderer IDs may map to one object. Missing IDs are errors, never
    silently treated as background. history records spatial/frame transforms.
    """

    dataset_id: str
    episode_id: str
    camera: str
    frame_ids: np.ndarray
    labels: np.ndarray  # [T,H,W], compact logical IDs
    objects: tuple[str, ...]
    history: tuple[dict, ...] = ()

    @property
    def stream_key(self):
        return self.dataset_id, self.episode_id, self.camera

    def validate(self):
        if any(not isinstance(v, str) or not v for v in self.stream_key):
            raise ValueError('dataset_id, episode_id and camera must be nonempty strings')
        if (not isinstance(self.objects, tuple) or not self.objects
                or any(not isinstance(v, str) or not v for v in self.objects)
                or self.objects[0] != 'background'
                or len(set(self.objects)) != len(self.objects)):
            raise ValueError('objects must be unique names with background first')
        if (self.labels.ndim != 3 or any(n == 0 for n in self.labels.shape)
                or self.labels.dtype.kind not in 'ui'):
            raise ValueError('labels must be a nonempty integer [T,H,W] array')
        if self.labels.min() < 0 or self.labels.max() >= len(self.objects):
            raise ValueError('label is outside the object table')
        if (self.frame_ids.shape != (len(self.labels),)
                or self.frame_ids.dtype.kind not in 'ui'
                or np.any(self.frame_ids < 0)
                or len(np.unique(self.frame_ids)) != len(self.frame_ids)):
            raise ValueError('frame IDs must be unique nonnegative integers, one per frame')
        if not isinstance(self.history, tuple) or any(not isinstance(v, dict) for v in self.history):
            raise ValueError('history must be a tuple of transform records')

    @classmethod
    def from_renderer(cls, raw_ids, *, renderer_objects, dataset_id, episode_id,
                      camera, frame_ids):
        """Remap potentially large renderer IDs to small episode-local IDs.

        renderer_objects must explicitly include all observed IDs, including
        background IDs. It may also include currently occluded objects.
        """
        raw_ids = np.asarray(raw_ids)
        if raw_ids.ndim != 3 or raw_ids.dtype.kind not in 'ui' or raw_ids.size == 0:
            raise ValueError('renderer IDs must be a nonempty integer [T,H,W] array')
        min_id = int(raw_ids.min())
        if min_id < 0:
            raise ValueError('negative renderer ID')
        if (not renderer_objects or any(
                not isinstance(k, (int, np.integer)) or k < 0
                or not isinstance(v, str) or not v
                for k, v in renderer_objects.items())):
            raise ValueError('renderer_objects must map nonnegative IDs to names')
        objects = ('background',) + tuple(sorted(set(renderer_objects.values()) - {'background'}))
        name_to_id = {name: i for i, name in enumerate(objects)}
        raw_ids = np.asarray(raw_ids)
        label_dtype = _label_dtype(len(objects))
        dense_limit = 1_000_000
        max_id = int(raw_ids.max())
        use_dense_lookup = min_id >= 0 and max_id <= dense_limit

        if use_dense_lookup:
            # Renderer actor IDs are usually small non-negative integers. A
            # dense lookup avoids np.unique(return_inverse=True), whose int64
            # inverse array can be several times larger than the label image.
            ids32 = raw_ids.astype(np.int32, copy=False)
            present = np.flatnonzero(
                np.bincount(ids32.reshape(-1), minlength=max_id + 1)
            )
            missing = set(map(int, present)) - set(renderer_objects)
            if missing:
                raise ValueError(f'unregistered renderer IDs: {sorted(missing)}')
            lookup = np.empty(max_id + 1, dtype=label_dtype)
            for renderer_id in present:
                lookup[renderer_id] = name_to_id[renderer_objects[int(renderer_id)]]
            labels = lookup[ids32]
        else:
            unique, inverse = np.unique(raw_ids, return_inverse=True)
            missing = set(map(int, unique)) - set(renderer_objects)
            if missing:
                raise ValueError(f'unregistered renderer IDs: {sorted(missing)}')
            lookup = np.asarray([name_to_id[renderer_objects[int(v)]] for v in unique],
                                dtype=label_dtype)
            labels = lookup[inverse].reshape(raw_ids.shape)

        result = cls(dataset_id, episode_id, camera, np.asarray(frame_ids).copy(),
                     labels, objects)
        result.validate()
        return result

    def mask(self, object_name):
        self.validate()
        return self.labels == self.objects.index(object_name)

    def save(self, path):
        """Atomic HDF5 sidecar; per-frame chunks, gzip, no content hashes."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            with h5py.File(temporary, 'w') as f:
                f.attrs['schema_version'] = SCHEMA_VERSION
                for field in ('dataset_id', 'episode_id', 'camera'):
                    f.attrs[field] = getattr(self, field)
                f.attrs['objects'] = json.dumps(self.objects, ensure_ascii=False)
                f.attrs['history'] = json.dumps(self.history, ensure_ascii=False)
                f.create_dataset('frame_ids', data=self.frame_ids)
                f.create_dataset('labels', data=self.labels.astype(_label_dtype(len(self.objects))),
                                 chunks=(1, *self.labels.shape[1:]), compression='gzip',
                                 compression_opts=1, shuffle=True)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        with h5py.File(path, 'r') as f:
            required = {'schema_version', 'dataset_id', 'episode_id', 'camera', 'objects', 'history'}
            if not required.issubset(f.attrs) or not {'frame_ids', 'labels'}.issubset(f):
                raise ValueError('missing annotation fields')
            if f.attrs['schema_version'] != SCHEMA_VERSION:
                raise ValueError('unsupported annotation schema')
            objects = json.loads(f.attrs['objects'])
            history = json.loads(f.attrs['history'])
            if not isinstance(objects, list) or not isinstance(history, list):
                raise ValueError('objects and history must be lists')
            result = cls(f.attrs['dataset_id'], f.attrs['episode_id'], f.attrs['camera'],
                         f['frame_ids'][:], f['labels'][:], tuple(objects), tuple(history))
        result.validate()
        return result


def transform_pair(rgb, annotations, *, stream_key, frame_ids,
                   indices=None, crop=None, size=None, flip_horizontal=False):
    """Validate and transform RGB and labels together; return (RGB, annotations).

    Pass ``rgb=None`` to transform labels only when the caller does not need a
    transformed RGB result. RGB: uint8 [T,H,W,3] when provided. crop:
    (left,top,right,bottom), with exclusive upper bounds. size: (width,height).
    Order: frame selection, crop, resize, horizontal flip. RGB uses bilinear
    resize; IDs use nearest at the same pixel centers.
    Input frame IDs must exactly match annotations, including order. Frame
    selection preserves original IDs; no timestamp/FPS guessing or hashing.
    """
    annotations.validate()
    if tuple(stream_key) != annotations.stream_key:
        raise ValueError('dataset/episode/camera mismatch')
    if not np.array_equal(np.asarray(frame_ids), annotations.frame_ids):
        raise ValueError('RGB and annotation frame IDs differ')
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (*annotations.labels.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('RGB must be uint8 [T,H,W,3] aligned with labels')
    labels = annotations.labels
    ids = annotations.frame_ids
    record = {}
    if indices is not None:
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in 'ui'
                or np.any(indices < 0) or np.any(indices >= len(ids))
                or len(np.unique(indices)) != len(indices)):
            raise ValueError('indices must be unique valid frame positions')
        if rgb is not None:
            rgb = rgb[indices]
        labels, ids = labels[indices], ids[indices]
        record['indices'] = indices.tolist()
    if crop is not None:
        if len(crop) != 4 or any(type(v) is not int for v in crop):
            raise ValueError('crop requires four integer coordinates')
        left, top, right, bottom = crop
        if not (0 <= left < right <= labels.shape[2] and 0 <= top < bottom <= labels.shape[1]):
            raise ValueError('crop outside image')
        if rgb is not None:
            rgb = rgb[:, top:bottom, left:right]
        labels = labels[:, top:bottom, left:right]
        record['crop'] = list(crop)
    if size is not None:
        if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('size must be positive integer (width,height)')
        width, height = size
        # Same half-pixel center convention as Pillow. Integer labels are never
        # passed through floating-point interpolation or Pillow signed mode I.
        ys = np.minimum(((np.arange(height) + .5) * labels.shape[1] / height).astype(int), labels.shape[1]-1)
        xs = np.minimum(((np.arange(width) + .5) * labels.shape[2] / width).astype(int), labels.shape[2]-1)
        labels = labels[:, ys[:, None], xs[None, :]]
        if rgb is not None:
            rgb = np.stack([np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)) for frame in rgb])
        record['size'] = list(size)
    if flip_horizontal:
        if rgb is not None:
            rgb = rgb[:, :, ::-1]
        labels = labels[:, :, ::-1]
        record['flip_horizontal'] = True
    output = replace(annotations, labels=labels.copy(), frame_ids=ids.copy(),
                     history=annotations.history + ((record,) if record else ()))
    output.validate()
    return (None if rgb is None else np.ascontiguousarray(rgb)), output

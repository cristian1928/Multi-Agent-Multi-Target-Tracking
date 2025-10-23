import csv
import os
from collections import defaultdict
from csv import DictWriter
from typing import TYPE_CHECKING, Any, Dict, List, TextIO

import numpy as np

if TYPE_CHECKING:
    from src.core.entity import Agent, Target
    CSVDictWriter = DictWriter[Any]
else:
    CSVDictWriter = DictWriter

DATA_DIR = 'simulation_data'
AGENT_DATA_DIR = os.path.join(DATA_DIR, 'agent_data')
TARGET_DATA_DIR = os.path.join(DATA_DIR, 'target_data')
STATE_DATA_SUFFIX = '_state_data.csv'
NN_DATA_SUFFIX = '_nn_data.csv'

_file_handles: Dict[str, TextIO] = {}
_csv_writers: Dict[str, CSVDictWriter] = {}
_data_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_buffer_size: int = 100


def ensure_directory_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def _get_csv_writer(file_path: str, headers: List[str], step: int) -> CSVDictWriter:
    if file_path not in _file_handles:
        if step == 1 and os.path.exists(file_path):
            os.remove(file_path)
        _file_handles[file_path] = open(file_path, 'w', newline='', buffering=8192)
        _csv_writers[file_path] = csv.DictWriter(_file_handles[file_path], fieldnames=headers)
        _csv_writers[file_path].writeheader()
    return _csv_writers[file_path]


def _flush_buffer(file_path: str) -> None:
    if file_path in _data_buffers and _data_buffers[file_path]:
        writer = _csv_writers[file_path]
        writer.writerows(_data_buffers[file_path])
        _file_handles[file_path].flush()
        _data_buffers[file_path].clear()


def save_state_to_csv(step: int, time: float, agents: List["Agent"], targets: List["Target"]) -> None:
    ensure_directory_exists(DATA_DIR)
    ensure_directory_exists(AGENT_DATA_DIR)
    ensure_directory_exists(TARGET_DATA_DIR)

    for index, agent in enumerate(agents):
        synchronization_error_norm = float(np.linalg.norm(agent.synchronization_error))
        agent_name = getattr(agent, 'id', None)
        state_file_path = os.path.join(AGENT_DATA_DIR, f'{agent_name}{STATE_DATA_SUFFIX}')
        headers = ['Time', 'Position X', 'Position Y', 'Position Z', 'Synchronization Error Norm']
        _get_csv_writer(state_file_path, headers, step)
        row_data: Dict[str, Any] = {
            'Time': time,
            'Position X': float(agent.positions[0, step - 1]),
            'Position Y': float(agent.positions[1, step - 1]),
            'Position Z': float(agent.positions[2, step - 1]),
            'Synchronization Error Norm': synchronization_error_norm,
        }
        _data_buffers[state_file_path].append(row_data)
        if len(_data_buffers[state_file_path]) >= _buffer_size:
            _flush_buffer(state_file_path)

    for index, target in enumerate(targets):
        target_name = getattr(target, 'id', None)
        state_file_path = os.path.join(TARGET_DATA_DIR, f'{target_name}{STATE_DATA_SUFFIX}')
        headers = ['Time', 'Position X', 'Position Y', 'Position Z']
        _get_csv_writer(state_file_path, headers, step)
        row_data: Dict[str, Any] = {
            'Time': time,
            'Position X': float(target.positions[0, step - 1]),
            'Position Y': float(target.positions[1, step - 1]),
            'Position Z': float(target.positions[2, step - 1]),
        }
        _data_buffers[state_file_path].append(row_data)
        if len(_data_buffers[state_file_path]) >= _buffer_size:
            _flush_buffer(state_file_path)


def close_all_files() -> None:
    for file_path in list(_data_buffers.keys()):
        _flush_buffer(file_path)
    for handle in _file_handles.values():
        handle.close()
    _file_handles.clear()
    _csv_writers.clear()
    _data_buffers.clear()

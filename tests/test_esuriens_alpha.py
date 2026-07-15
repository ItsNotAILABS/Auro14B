"""Alpha tests for the ESURIENS task storage system.

100 tests covering:
- TaskHorizon data model (creation, serialization, state transitions)
- HorizonLevel classification
- MemoryTemple persistence (save/load/archive/verify)
- SovereignArtifact and TaskChain structures
- EsuriensTaskStorage hybrid volatile/persistent architecture
- Cross-restart persistence
- Task chain archival workflow
"""

import json
import os
import tempfile
import time

import pytest

from mesie.esuriens.task_horizon import HorizonLevel, TaskHorizon
from mesie.esuriens.task_storage import EsuriensTaskStorage
from mesie.esuriens.memory_temple import MemoryTemple
from mesie.esuriens.sovereign_artifact import SovereignArtifact, TaskChain


@pytest.fixture
def temple_dir(tmp_path):
    """Provide a temp directory for Memory Temple storage."""
    return tmp_path / "memory_temple"


@pytest.fixture
def temple(temple_dir):
    """Provide a fresh MemoryTemple instance."""
    return MemoryTemple(temple_dir)


@pytest.fixture
def storage(temple):
    """Provide a fresh EsuriensTaskStorage instance."""
    return EsuriensTaskStorage(temple)


# ═══════════════════════════════════════════════════════════════════
# Tests 1-10: HorizonLevel enum
# ═══════════════════════════════════════════════════════════════════


class TestHorizonLevel:
    def test_01_immediate_value(self):
        assert HorizonLevel.IMMEDIATE.value == "immediate"

    def test_02_short_term_value(self):
        assert HorizonLevel.SHORT_TERM.value == "short_term"

    def test_03_long_term_value(self):
        assert HorizonLevel.LONG_TERM.value == "long_term"

    def test_04_from_string_immediate(self):
        assert HorizonLevel("immediate") == HorizonLevel.IMMEDIATE

    def test_05_from_string_short_term(self):
        assert HorizonLevel("short_term") == HorizonLevel.SHORT_TERM

    def test_06_from_string_long_term(self):
        assert HorizonLevel("long_term") == HorizonLevel.LONG_TERM

    def test_07_invalid_raises(self):
        with pytest.raises(ValueError):
            HorizonLevel("invalid")

    def test_08_all_levels_count(self):
        assert len(HorizonLevel) == 3

    def test_09_unique_values(self):
        values = [h.value for h in HorizonLevel]
        assert len(values) == len(set(values))

    def test_10_enum_membership(self):
        assert HorizonLevel.IMMEDIATE in HorizonLevel


# ═══════════════════════════════════════════════════════════════════
# Tests 11-30: TaskHorizon data model
# ═══════════════════════════════════════════════════════════════════


class TestTaskHorizon:
    def test_11_default_creation(self):
        task = TaskHorizon()
        assert task.task_id is not None
        assert task.status == "pending"

    def test_12_custom_title(self):
        task = TaskHorizon(title="Test Task")
        assert task.title == "Test Task"

    def test_13_default_horizon_is_immediate(self):
        task = TaskHorizon()
        assert task.horizon == HorizonLevel.IMMEDIATE

    def test_14_set_short_term(self):
        task = TaskHorizon(horizon=HorizonLevel.SHORT_TERM)
        assert task.horizon == HorizonLevel.SHORT_TERM

    def test_15_set_long_term(self):
        task = TaskHorizon(horizon=HorizonLevel.LONG_TERM)
        assert task.horizon == HorizonLevel.LONG_TERM

    def test_16_complete_changes_status(self):
        task = TaskHorizon()
        task.complete()
        assert task.status == "completed"

    def test_17_complete_sets_timestamp(self):
        task = TaskHorizon()
        task.complete()
        assert task.completed_at is not None

    def test_18_fail_changes_status(self):
        task = TaskHorizon()
        task.fail()
        assert task.status == "failed"

    def test_19_activate_changes_status(self):
        task = TaskHorizon()
        task.activate()
        assert task.status == "active"

    def test_20_archive_changes_status(self):
        task = TaskHorizon()
        task.archive()
        assert task.status == "archived"

    def test_21_is_terminal_completed(self):
        task = TaskHorizon()
        task.complete()
        assert task.is_terminal is True

    def test_22_is_terminal_failed(self):
        task = TaskHorizon()
        task.fail()
        assert task.is_terminal is True

    def test_23_is_terminal_pending(self):
        task = TaskHorizon()
        assert task.is_terminal is False

    def test_24_is_active_pending(self):
        task = TaskHorizon()
        assert task.is_active is True

    def test_25_is_active_active(self):
        task = TaskHorizon()
        task.activate()
        assert task.is_active is True

    def test_26_is_active_completed(self):
        task = TaskHorizon()
        task.complete()
        assert task.is_active is False

    def test_27_to_dict_has_all_fields(self):
        task = TaskHorizon(title="T", description="D")
        d = task.to_dict()
        assert "task_id" in d
        assert d["title"] == "T"
        assert d["description"] == "D"
        assert d["horizon"] == "immediate"

    def test_28_from_dict_roundtrip(self):
        task = TaskHorizon(title="RT", priority=5, horizon=HorizonLevel.LONG_TERM)
        restored = TaskHorizon.from_dict(task.to_dict())
        assert restored.title == "RT"
        assert restored.priority == 5
        assert restored.horizon == HorizonLevel.LONG_TERM

    def test_29_dependencies_list(self):
        task = TaskHorizon(dependencies=["a", "b", "c"])
        assert len(task.dependencies) == 3

    def test_30_spectral_signature(self):
        task = TaskHorizon(spectral_signature=[1.0, 2.0, 3.0])
        assert task.spectral_signature == [1.0, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════════
# Tests 31-45: TaskChain and SovereignArtifact
# ═══════════════════════════════════════════════════════════════════


class TestTaskChainAndArtifact:
    def test_31_chain_default_creation(self):
        chain = TaskChain()
        assert chain.chain_id is not None
        assert chain.task_ids == []

    def test_32_chain_with_tasks(self):
        chain = TaskChain(task_ids=["t1", "t2", "t3"])
        assert len(chain.task_ids) == 3

    def test_33_chain_root_task(self):
        chain = TaskChain(root_task_id="root1")
        assert chain.root_task_id == "root1"

    def test_34_chain_to_dict(self):
        chain = TaskChain(task_ids=["a", "b"])
        d = chain.to_dict()
        assert d["task_ids"] == ["a", "b"]

    def test_35_chain_from_dict_roundtrip(self):
        chain = TaskChain(task_ids=["x"], horizon_levels=["immediate", "long_term"])
        restored = TaskChain.from_dict(chain.to_dict())
        assert restored.task_ids == ["x"]
        assert "immediate" in restored.horizon_levels

    def test_36_artifact_default_creation(self):
        art = SovereignArtifact()
        assert art.artifact_id is not None
        assert art.sovereignty_level == 0

    def test_37_artifact_with_chain(self):
        chain = TaskChain(task_ids=["t1"])
        art = SovereignArtifact(chain=chain)
        assert art.chain.task_ids == ["t1"]

    def test_38_artifact_sovereignty_levels(self):
        art = SovereignArtifact(sovereignty_level=2)
        assert art.sovereignty_level == 2

    def test_39_artifact_insights(self):
        art = SovereignArtifact(insights=["learned X", "discovered Y"])
        assert len(art.insights) == 2

    def test_40_artifact_to_dict(self):
        art = SovereignArtifact(sovereignty_level=1)
        d = art.to_dict()
        assert d["sovereignty_level"] == 1
        assert "chain" in d

    def test_41_artifact_from_dict_roundtrip(self):
        chain = TaskChain(task_ids=["a", "b"])
        art = SovereignArtifact(chain=chain, sovereignty_level=2)
        restored = SovereignArtifact.from_dict(art.to_dict())
        assert restored.sovereignty_level == 2
        assert restored.chain.task_ids == ["a", "b"]

    def test_42_artifact_spectral_fingerprint(self):
        art = SovereignArtifact(spectral_fingerprint=[0.1, 0.2, 0.3])
        assert art.spectral_fingerprint == [0.1, 0.2, 0.3]

    def test_43_chain_duration(self):
        chain = TaskChain(total_duration=42.5)
        assert chain.total_duration == 42.5

    def test_44_chain_horizon_levels_list(self):
        chain = TaskChain(horizon_levels=["immediate", "short_term", "long_term"])
        assert len(chain.horizon_levels) == 3

    def test_45_artifact_digest_initially_empty(self):
        art = SovereignArtifact()
        assert art.digest == ""


# ═══════════════════════════════════════════════════════════════════
# Tests 46-65: MemoryTemple persistence
# ═══════════════════════════════════════════════════════════════════


class TestMemoryTemple:
    def test_46_temple_creates_directory(self, temple_dir):
        temple = MemoryTemple(temple_dir)
        assert temple_dir.exists()

    def test_47_temple_creates_artifacts_dir(self, temple):
        assert (temple.temple_path / "sovereign_artifacts").is_dir()

    def test_48_save_empty_horizons(self, temple):
        temple.save_horizons([])
        loaded = temple.load_horizons()
        assert loaded == []

    def test_49_save_and_load_single_task(self, temple):
        task = TaskHorizon(title="Persist me")
        temple.save_horizons([task])
        loaded = temple.load_horizons()
        assert len(loaded) == 1
        assert loaded[0].title == "Persist me"

    def test_50_save_and_load_multiple_tasks(self, temple):
        tasks = [
            TaskHorizon(title=f"Task {i}", horizon=HorizonLevel.SHORT_TERM)
            for i in range(5)
        ]
        temple.save_horizons(tasks)
        loaded = temple.load_horizons()
        assert len(loaded) == 5

    def test_51_load_nonexistent_returns_empty(self, temple_dir):
        temple = MemoryTemple(temple_dir / "fresh")
        assert temple.load_horizons() == []

    def test_52_archive_artifact_creates_file(self, temple):
        art = SovereignArtifact()
        path = temple.archive_artifact(art)
        assert os.path.exists(path)

    def test_53_archive_sets_digest(self, temple):
        art = SovereignArtifact()
        temple.archive_artifact(art)
        assert art.digest != ""

    def test_54_load_artifact_by_id(self, temple):
        art = SovereignArtifact(sovereignty_level=1)
        temple.archive_artifact(art)
        loaded = temple.load_artifact(art.artifact_id)
        assert loaded is not None
        assert loaded.sovereignty_level == 1

    def test_55_load_nonexistent_artifact(self, temple):
        assert temple.load_artifact("nonexistent") is None

    def test_56_list_artifacts_empty(self, temple):
        assert temple.list_artifacts() == []

    def test_57_list_artifacts_after_archive(self, temple):
        art = SovereignArtifact()
        temple.archive_artifact(art)
        ids = temple.list_artifacts()
        assert art.artifact_id in ids

    def test_58_list_multiple_artifacts(self, temple):
        for _ in range(3):
            temple.archive_artifact(SovereignArtifact())
        assert len(temple.list_artifacts()) == 3

    def test_59_verify_integrity_valid(self, temple):
        art = SovereignArtifact()
        temple.archive_artifact(art)
        assert temple.verify_artifact_integrity(art.artifact_id) is True

    def test_60_verify_integrity_nonexistent(self, temple):
        assert temple.verify_artifact_integrity("nope") is False

    def test_61_clear_removes_horizons(self, temple):
        temple.save_horizons([TaskHorizon(title="gone")])
        temple.clear()
        assert temple.load_horizons() == []

    def test_62_clear_removes_artifacts(self, temple):
        temple.archive_artifact(SovereignArtifact())
        temple.clear()
        assert temple.list_artifacts() == []

    def test_63_persist_across_instances(self, temple_dir):
        temple1 = MemoryTemple(temple_dir)
        task = TaskHorizon(title="Survivor")
        temple1.save_horizons([task])
        temple2 = MemoryTemple(temple_dir)
        loaded = temple2.load_horizons()
        assert len(loaded) == 1
        assert loaded[0].title == "Survivor"

    def test_64_artifact_persist_across_instances(self, temple_dir):
        temple1 = MemoryTemple(temple_dir)
        art = SovereignArtifact(sovereignty_level=2)
        temple1.archive_artifact(art)
        temple2 = MemoryTemple(temple_dir)
        loaded = temple2.load_artifact(art.artifact_id)
        assert loaded is not None
        assert loaded.sovereignty_level == 2

    def test_65_horizon_file_is_valid_json(self, temple):
        temple.save_horizons([TaskHorizon(title="Check")])
        path = temple.temple_path / "task_horizons.json"
        data = json.loads(path.read_text())
        assert "tasks" in data
        assert data["version"] == 1


# ═══════════════════════════════════════════════════════════════════
# Tests 66-100: EsuriensTaskStorage hybrid system
# ═══════════════════════════════════════════════════════════════════


class TestEsuriensTaskStorage:
    def test_66_initial_empty(self, storage):
        assert storage.task_count == 0

    def test_67_add_task(self, storage):
        tid = storage.add_task(TaskHorizon(title="New"))
        assert tid is not None
        assert storage.task_count == 1

    def test_68_get_task(self, storage):
        task = TaskHorizon(title="Find me")
        storage.add_task(task)
        found = storage.get_task(task.task_id)
        assert found is not None
        assert found.title == "Find me"

    def test_69_get_nonexistent(self, storage):
        assert storage.get_task("nope") is None

    def test_70_remove_task(self, storage):
        task = TaskHorizon()
        storage.add_task(task)
        assert storage.remove_task(task.task_id) is True
        assert storage.task_count == 0

    def test_71_remove_nonexistent(self, storage):
        assert storage.remove_task("nope") is False

    def test_72_complete_task(self, storage):
        task = TaskHorizon()
        storage.add_task(task)
        assert storage.complete_task(task.task_id) is True
        assert storage.get_task(task.task_id).status == "completed"

    def test_73_fail_task(self, storage):
        task = TaskHorizon()
        storage.add_task(task)
        assert storage.fail_task(task.task_id) is True
        assert storage.get_task(task.task_id).status == "failed"

    def test_74_activate_task(self, storage):
        task = TaskHorizon()
        storage.add_task(task)
        assert storage.activate_task(task.task_id) is True
        assert storage.get_task(task.task_id).status == "active"

    def test_75_complete_nonexistent(self, storage):
        assert storage.complete_task("nope") is False

    def test_76_get_by_horizon_immediate(self, storage):
        storage.add_task(TaskHorizon(horizon=HorizonLevel.IMMEDIATE))
        storage.add_task(TaskHorizon(horizon=HorizonLevel.LONG_TERM))
        assert len(storage.get_immediate()) == 1

    def test_77_get_by_horizon_short_term(self, storage):
        storage.add_task(TaskHorizon(horizon=HorizonLevel.SHORT_TERM))
        assert len(storage.get_short_term()) == 1

    def test_78_get_by_horizon_long_term(self, storage):
        storage.add_task(TaskHorizon(horizon=HorizonLevel.LONG_TERM))
        storage.add_task(TaskHorizon(horizon=HorizonLevel.LONG_TERM))
        assert len(storage.get_long_term()) == 2

    def test_79_get_active_tasks(self, storage):
        t1 = TaskHorizon()
        t2 = TaskHorizon()
        storage.add_task(t1)
        storage.add_task(t2)
        storage.complete_task(t2.task_id)
        assert len(storage.get_active_tasks()) == 1

    def test_80_get_completed_tasks(self, storage):
        t = TaskHorizon()
        storage.add_task(t)
        storage.complete_task(t.task_id)
        assert len(storage.get_completed_tasks()) == 1

    def test_81_archive_chain(self, storage):
        tasks = [TaskHorizon(title=f"Chain {i}") for i in range(3)]
        for t in tasks:
            storage.add_task(t)
            storage.complete_task(t.task_id)
        ids = [t.task_id for t in tasks]
        art = storage.archive_completed_chain(ids)
        assert art is not None
        assert len(art.chain.task_ids) == 3

    def test_82_archive_removes_from_volatile(self, storage):
        t = TaskHorizon()
        storage.add_task(t)
        storage.complete_task(t.task_id)
        storage.archive_completed_chain([t.task_id])
        assert storage.get_task(t.task_id) is None

    def test_83_archive_persists_to_temple(self, storage, temple):
        t = TaskHorizon()
        storage.add_task(t)
        storage.complete_task(t.task_id)
        art = storage.archive_completed_chain([t.task_id])
        assert art.artifact_id in temple.list_artifacts()

    def test_84_archive_incomplete_fails(self, storage):
        t = TaskHorizon()
        storage.add_task(t)
        assert storage.archive_completed_chain([t.task_id]) is None

    def test_85_archive_nonexistent_fails(self, storage):
        assert storage.archive_completed_chain(["fake"]) is None

    def test_86_flush_completed(self, storage):
        tasks = [TaskHorizon() for _ in range(3)]
        for t in tasks:
            storage.add_task(t)
            storage.complete_task(t.task_id)
        arts = storage.flush_completed()
        assert len(arts) == 3

    def test_87_flush_empty(self, storage):
        assert storage.flush_completed() == []

    def test_88_snapshot(self, storage):
        storage.add_task(TaskHorizon(horizon=HorizonLevel.IMMEDIATE))
        storage.add_task(TaskHorizon(horizon=HorizonLevel.LONG_TERM))
        snap = storage.snapshot()
        assert snap["total_tasks"] == 2
        assert snap["immediate"] == 1
        assert snap["long_term"] == 1

    def test_89_restart_persistence(self, temple_dir):
        """Test that tasks persist across storage restarts."""
        temple = MemoryTemple(temple_dir)
        storage1 = EsuriensTaskStorage(temple)
        task = TaskHorizon(title="Persist across restart")
        storage1.add_task(task)

        # Simulate restart — new storage, same temple
        storage2 = EsuriensTaskStorage(MemoryTemple(temple_dir))
        found = storage2.get_task(task.task_id)
        assert found is not None
        assert found.title == "Persist across restart"

    def test_90_restart_preserves_horizon(self, temple_dir):
        temple = MemoryTemple(temple_dir)
        storage1 = EsuriensTaskStorage(temple)
        storage1.add_task(TaskHorizon(horizon=HorizonLevel.LONG_TERM, title="LT"))

        storage2 = EsuriensTaskStorage(MemoryTemple(temple_dir))
        assert len(storage2.get_long_term()) == 1

    def test_91_completed_not_restored(self, temple_dir):
        """Completed tasks are not restored (they should be archived)."""
        temple = MemoryTemple(temple_dir)
        storage1 = EsuriensTaskStorage(temple)
        t = TaskHorizon()
        storage1.add_task(t)
        storage1.complete_task(t.task_id)

        storage2 = EsuriensTaskStorage(MemoryTemple(temple_dir))
        # Completed tasks don't get restored to volatile
        assert storage2.get_task(t.task_id) is None

    def test_92_sovereignty_level_scaling(self, storage):
        """Sovereignty level scales with chain length."""
        tasks = [TaskHorizon() for _ in range(6)]
        for t in tasks:
            storage.add_task(t)
            storage.complete_task(t.task_id)
        art = storage.archive_completed_chain([t.task_id for t in tasks])
        assert art.sovereignty_level == 2  # min(6//3, 2) = 2

    def test_93_short_chain_sovereignty(self, storage):
        t = TaskHorizon()
        storage.add_task(t)
        storage.complete_task(t.task_id)
        art = storage.archive_completed_chain([t.task_id])
        assert art.sovereignty_level == 0  # min(1//3, 2) = 0

    def test_94_chain_records_duration(self, storage):
        t = TaskHorizon()
        storage.add_task(t)
        storage.complete_task(t.task_id)
        art = storage.archive_completed_chain([t.task_id])
        assert art.chain.total_duration >= 0

    def test_95_chain_records_horizon_levels(self, storage):
        t1 = TaskHorizon(horizon=HorizonLevel.IMMEDIATE)
        t2 = TaskHorizon(horizon=HorizonLevel.LONG_TERM)
        storage.add_task(t1)
        storage.add_task(t2)
        storage.complete_task(t1.task_id)
        storage.complete_task(t2.task_id)
        art = storage.archive_completed_chain([t1.task_id, t2.task_id])
        assert "immediate" in art.chain.horizon_levels
        assert "long_term" in art.chain.horizon_levels

    def test_96_multiple_horizon_tasks(self, storage):
        for level in HorizonLevel:
            for _ in range(3):
                storage.add_task(TaskHorizon(horizon=level))
        assert storage.task_count == 9

    def test_97_auto_persist_disabled(self, temple):
        storage = EsuriensTaskStorage(temple, auto_persist=False)
        storage.add_task(TaskHorizon(title="Volatile only"))
        # Without auto-persist, horizons file shouldn't exist
        loaded = temple.load_horizons()
        assert len(loaded) == 0

    def test_98_priority_ordering(self, storage):
        storage.add_task(TaskHorizon(title="Low", priority=1))
        storage.add_task(TaskHorizon(title="High", priority=10))
        tasks = sorted(storage.get_active_tasks(), key=lambda t: t.priority, reverse=True)
        assert tasks[0].title == "High"

    def test_99_metadata_persistence(self, temple_dir):
        temple = MemoryTemple(temple_dir)
        storage = EsuriensTaskStorage(temple)
        t = TaskHorizon(metadata={"key": "value", "num": 42})
        storage.add_task(t)

        storage2 = EsuriensTaskStorage(MemoryTemple(temple_dir))
        found = storage2.get_task(t.task_id)
        assert found.metadata["key"] == "value"
        assert found.metadata["num"] == 42

    def test_100_full_lifecycle(self, temple_dir):
        """Full lifecycle: create -> activate -> complete -> archive -> verify."""
        temple = MemoryTemple(temple_dir)
        storage = EsuriensTaskStorage(temple)

        # Create tasks across horizons
        t1 = TaskHorizon(title="Immediate goal", horizon=HorizonLevel.IMMEDIATE)
        t2 = TaskHorizon(title="Short-term goal", horizon=HorizonLevel.SHORT_TERM, parent_id=t1.task_id)
        t3 = TaskHorizon(title="Long-term goal", horizon=HorizonLevel.LONG_TERM, parent_id=t2.task_id)

        for t in [t1, t2, t3]:
            storage.add_task(t)

        # Activate and complete chain
        for t in [t1, t2, t3]:
            storage.activate_task(t.task_id)
            storage.complete_task(t.task_id)

        # Archive the chain
        art = storage.archive_completed_chain([t1.task_id, t2.task_id, t3.task_id])
        assert art is not None
        assert art.chain.root_task_id == t1.task_id
        assert len(art.chain.horizon_levels) == 3

        # Verify persistence
        assert temple.verify_artifact_integrity(art.artifact_id) is True

        # Verify volatile is clean
        assert storage.task_count == 0

        # Verify artifact survives restart
        temple2 = MemoryTemple(temple_dir)
        loaded = temple2.load_artifact(art.artifact_id)
        assert loaded is not None
        assert loaded.chain.task_ids == [t1.task_id, t2.task_id, t3.task_id]

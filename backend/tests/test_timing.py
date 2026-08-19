from app.models.pipeline import StageTimings
from app.timing import stage_timer


def test_stage_timer_records_duration():
    timings = StageTimings()
    with stage_timer(timings, "encode"):
        pass
    assert "encode" in timings.stages
    assert timings.stages["encode"] >= 0


def test_stub_pipeline_records_multiple_stages():
    timings = StageTimings()
    for stage in ("encode", "retrieve", "generate"):
        with stage_timer(timings, stage):
            pass
    assert set(timings.stages) == {"encode", "retrieve", "generate"}
    assert timings.total_ms >= 0

import math

from compare.llumnix.policy import (
    InstanceState,
    LlumnixPolicyConfig,
    LlumnixQueuePolicy,
    RequestState,
)


def test_virtual_usage_counts_only_head_of_line_queued_demand():
    policy = LlumnixQueuePolicy()
    first = RequestState("first", "batch", 100, 100, queued=True, head_of_line=True)
    second = RequestState("second", "batch", 100, 100, queued=True, head_of_line=False)
    instance = InstanceState("a", 1000, [first, second])
    assert policy.virtual_usage(first, instance) == 200
    assert policy.virtual_usage(second, instance) == 0
    assert policy.freeness(instance) == 800


def test_interactive_request_receives_execution_headroom():
    policy = LlumnixQueuePolicy(LlumnixPolicyConfig(priority_headroom_tokens=400))
    request = RequestState("high", "interactive", 10, 10, physical_tokens=100, queued=False)
    instance = InstanceState("a", 1000, [request], active_requests=1)
    assert policy.virtual_usage(request, instance) == 500


def test_dispatch_uses_highest_freeness_and_migration_pairs_extremes():
    config = LlumnixPolicyConfig(migrate_out_freeness=100, migrate_in_freeness=500)
    policy = LlumnixQueuePolicy(config)
    overloaded_req = RequestState("r", "batch", 950, 0, queued=False, physical_tokens=950)
    overloaded = InstanceState("over", 1000, [overloaded_req], active_requests=1)
    free = InstanceState("free", 1000)
    assert policy.dispatch([overloaded, free]).instance_id == "free"
    assert policy.migration_pairs([overloaded, free]) == [("over", "free")]


def test_terminating_instance_has_negative_infinite_freeness():
    policy = LlumnixQueuePolicy()
    instance = InstanceState("a", 1000, terminating=True)
    assert policy.freeness(instance) == -math.inf


def test_migration_prefers_normal_short_request():
    source = InstanceState("source", 1000, requests=[
        RequestState("interactive", "interactive", 1, 1),
        RequestState("long", "batch", 100, 100),
        RequestState("short", "batch", 10, 10),
    ])
    selected = LlumnixQueuePolicy.select_request_to_migrate(source)
    assert selected.request_id == "short"


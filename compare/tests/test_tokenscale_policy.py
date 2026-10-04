from compare.tokenscale.policy import (
    Arrival,
    InstanceSnapshot,
    ROLE_BATCH,
    ROLE_INTERACTIVE,
    ROLE_MIXED,
    TokenScaleConfig,
    TokenScalePolicy,
    VelocityProfile,
)


def make_policy(**config_overrides):
    profile = VelocityProfile(
        prefill_tokens_per_s=100.0,
        convertible_prefill_tokens_per_s=50.0,
        decode_tokens_per_s={"S-S": 100.0, "default": 100.0},
    )
    config = TokenScaleConfig(
        arrival_window_s=2.0,
        min_interactive=1,
        min_batch=1,
        convertible_instances=1,
        max_instances=8,
        scale_down_hold_intervals=2,
        **config_overrides,
    )
    return TokenScalePolicy(profile, config)


def test_token_velocity_scales_by_tokens_not_request_count():
    policy = make_policy()
    policy.observe(Arrival(10.0, "interactive", 400, 10))
    plan = policy.plan(10.0)
    assert plan.interactive == 3  # ceil(1.15 * 200 input tok/s / 100 velocity)
    assert plan.mixed == 1


def test_bucketed_decoder_capacity_is_summed():
    policy = make_policy()
    for index in range(4):
        policy.observe(Arrival(10.0 + index * 0.01, "batch", 100, 100))
    plan = policy.plan(10.1)
    assert plan.batch >= 4


def test_convertible_instance_absorbs_interactive_burst_when_regular_misses_slo():
    policy = make_policy(interactive_ttft_slo_ms=200.0)
    instances = {
        "regular": InstanceSnapshot("regular", ROLE_INTERACTIVE, inflight_input_tokens=1000),
        "mixed": InstanceSnapshot("mixed", ROLE_MIXED),
        "batch": InstanceSnapshot("batch", ROLE_BATCH),
    }
    chosen = policy.choose_instance("interactive", 10, 20, instances, burst=True)
    assert chosen.instance_id == "mixed"


def test_role_assignment_preserves_existing_roles():
    policy = make_policy()
    policy.observe(Arrival(10.0, "interactive", 50, 10))
    plan = policy.plan(10.0)
    instances = [
        InstanceSnapshot("i", ROLE_INTERACTIVE),
        InstanceSnapshot("b", ROLE_BATCH),
        InstanceSnapshot("m", ROLE_MIXED),
        InstanceSnapshot("w"),
    ]
    assignments = policy.assign_roles(instances, plan)
    assert assignments["i"] == ROLE_INTERACTIVE
    assert assignments["b"] == ROLE_BATCH
    assert assignments["m"] == ROLE_MIXED


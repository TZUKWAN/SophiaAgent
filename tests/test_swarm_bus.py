from concurrent.futures import ThreadPoolExecutor

from sophia.swarm.bus import SwarmBus


def test_bus_write_read_and_latest():
    bus = SwarmBus()
    bus.register_agent("a1")
    bus.write("a1", "hello", msg_type="status")
    assert bus.read("a1")[0].content == "hello"
    assert bus.read_latest("a1")[0].msg_type == "status"


def test_bus_broadcast_reaches_all_channels():
    bus = SwarmBus()
    bus.register_agent("a1")
    bus.register_agent("a2")
    bus.broadcast("a1", "shared")
    assert bus.read("a1")[0].content == "shared"
    assert bus.read("a2")[0].metadata["broadcast"] is True


def test_bus_truncates_channels_and_context():
    bus = SwarmBus(max_messages_per_channel=2)
    bus.register_agent("a1")
    for idx in range(4):
        bus.write("a1", f"msg-{idx}")
    assert len(bus.read("a1")) == 2
    assert "[truncated]" in bus.to_context_string(max_length=20)


def test_bus_thread_safe_writes():
    bus = SwarmBus()
    bus.register_agent("a1")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: bus.write("a1", str(i)), range(20)))
    assert len(bus.read("a1")) == 20

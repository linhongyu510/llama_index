"""
Tests for ref_doc_info cleanup when a node's SOURCE relationship changes.

Regression test for bug: when a node's ref_doc_id changes via add_documents(allow_update=True),
the old ref_doc_info still contains the node_id, leading to incorrect deletion when
delete_ref_doc is called on the old ref.
"""

import pytest
import json
from typing import List, Tuple
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.kvstore import SimpleKVStore


def test_update_node_ref_doc_id_cleans_up_old_ref() -> None:
    """
    When a node's SOURCE ref_doc_id changes, old ref_doc_info should not retain the node_id.

    This prevents delete_ref_doc(old_ref) from incorrectly deleting a node that
    now belongs to a new ref.
    """
    store = SimpleDocumentStore()

    # Add node under ref_A
    node_a = TextNode(
        text="Node content",
        id_="movable_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node_a])

    # Update same node_id with new SOURCE = ref_B
    node_b = TextNode(
        text="Node content",
        id_="movable_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    store.add_documents([node_b], allow_update=True)

    # Old ref should no longer have the node
    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is None or "movable_node" not in ref_a_info.node_ids, (
        "Old ref_doc_info should not contain the moved node_id"
    )

    # New ref should have the node
    ref_b_info = store.get_ref_doc_info("ref_B")
    assert ref_b_info is not None and "movable_node" in ref_b_info.node_ids, (
        "New ref_doc_info should contain the moved node_id"
    )

    # Deleting old ref should not delete the moved node
    store.delete_ref_doc("ref_A")
    assert store.document_exists("movable_node"), (
        "Node that moved to ref_B should survive delete_ref_doc(ref_A)"
    )


def test_update_node_no_source_cleans_up_old_ref() -> None:
    """When a node's SOURCE is removed, old ref_doc_info should not retain the node_id."""
    store = SimpleDocumentStore()

    node_a = TextNode(
        text="Node content",
        id_="nosrc_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node_a])

    # Update with no SOURCE relationship
    node_nosrc = TextNode(text="Node content", id_="nosrc_node")
    store.add_documents([node_nosrc], allow_update=True)

    # Old ref should no longer have the node
    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is None or "nosrc_node" not in ref_a_info.node_ids, (
        "Old ref_doc_info should not contain the node after SOURCE is removed"
    )

    # Deleting old ref should not delete the node
    store.delete_ref_doc("ref_A")
    assert store.document_exists("nosrc_node"), (
        "Node without SOURCE should survive delete_ref_doc(ref_A)"
    )


def test_update_node_ref_doc_id_subset_others_remain() -> None:
    """When one node moves from ref_A to ref_B, other nodes in ref_A should remain."""
    store = SimpleDocumentStore()

    node_stay = TextNode(
        text="Stay node",
        id_="stay_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    node_move = TextNode(
        text="Move node",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node_stay, node_move])

    # Move only move_node to ref_B
    node_move_b = TextNode(
        text="Move node",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    store.add_documents([node_move_b], allow_update=True)

    # ref_A should have stay_node but not move_node
    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is not None
    assert "stay_node" in ref_a_info.node_ids, "stay_node should remain in ref_A"
    assert "move_node" not in ref_a_info.node_ids, (
        "move_node should be removed from ref_A"
    )

    # ref_B should have move_node
    ref_b_info = store.get_ref_doc_info("ref_B")
    assert ref_b_info is not None and "move_node" in ref_b_info.node_ids

    # Deleting ref_A should delete stay_node but not move_node
    store.delete_ref_doc("ref_A")
    assert not store.document_exists("stay_node"), "stay_node should be deleted"
    assert store.document_exists("move_node"), "move_node should survive"


@pytest.mark.asyncio
async def test_async_update_node_ref_doc_id_cleans_up_old_ref() -> None:
    """Async version: when a node's SOURCE ref_doc_id changes, old ref_doc_info should be cleaned up."""
    store = SimpleDocumentStore()

    node_a = TextNode(
        text="Async node",
        id_="async_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    await store.async_add_documents([node_a])

    node_b = TextNode(
        text="Async node",
        id_="async_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    await store.async_add_documents([node_b], allow_update=True)

    # Deleting old ref should not delete the moved node
    await store.adelete_ref_doc("ref_A")
    assert store.document_exists("async_node"), (
        "Async: node that moved to ref_B should survive delete_ref_doc(ref_A)"
    )


def test_update_node_same_ref_no_duplicate() -> None:
    """When a node is updated with the same ref_doc_id, it should not be duplicated in node_ids."""
    store = SimpleDocumentStore()

    node = TextNode(
        text="Original",
        id_="same_ref_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node])

    # Update with same ref
    node_updated = TextNode(
        text="Updated",
        id_="same_ref_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node_updated], allow_update=True)

    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is not None
    assert ref_a_info.node_ids.count("same_ref_node") == 1, (
        "Same ref update should not duplicate node_id"
    )


def test_batch_interleaved_move_and_stay() -> None:
    """
    Batch: one node moves A→B, another stays in A.

    The cleanup should not remove the stay-node from ref_A, and should
    correctly remove the moved node from ref_A.
    """
    store = SimpleDocumentStore()

    # Initial state: both nodes under ref_A
    node_stay = TextNode(
        text="Stay",
        id_="stay_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    node_move = TextNode(
        text="Move",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node_stay, node_move])

    # Batch update: move_node → ref_B, stay_node stays in ref_A
    node_move_b = TextNode(
        text="Move updated",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    node_stay_b = TextNode(
        text="Stay updated",
        id_="stay_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node_move_b, node_stay_b], allow_update=True)

    # ref_A should have stay_node but not move_node
    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is not None
    assert "stay_node" in ref_a_info.node_ids, "stay_node should remain in ref_A"
    assert "move_node" not in ref_a_info.node_ids, (
        "move_node should be removed from ref_A"
    )

    # ref_B should have move_node
    ref_b_info = store.get_ref_doc_info("ref_B")
    assert ref_b_info is not None and "move_node" in ref_b_info.node_ids


def test_batch_swap_refs() -> None:
    """Batch: node1 A→B and node2 B→A (swap)."""
    store = SimpleDocumentStore()

    # Initial: node1 under A, node2 under B
    node1 = TextNode(
        text="Node1",
        id_="node1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    node2 = TextNode(
        text="Node2",
        id_="node2",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    store.add_documents([node1, node2])

    # Swap: node1 → B, node2 → A
    node1_b = TextNode(
        text="Node1 updated",
        id_="node1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    node2_a = TextNode(
        text="Node2 updated",
        id_="node2",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node1_b, node2_a], allow_update=True)

    # After swap: ref_A has node2, ref_B has node1
    ref_a_info = store.get_ref_doc_info("ref_A")
    ref_b_info = store.get_ref_doc_info("ref_B")
    assert ref_a_info is not None
    assert "node2" in ref_a_info.node_ids, "node2 should be in ref_A after swap"
    assert "node1" not in ref_a_info.node_ids, "node1 should not be in ref_A after swap"
    assert ref_b_info is not None
    assert "node1" in ref_b_info.node_ids, "node1 should be in ref_B after swap"
    assert "node2" not in ref_b_info.node_ids, "node2 should not be in ref_B after swap"


def test_allow_update_false_no_mutate_ref_info() -> None:
    """
    When allow_update=False and a duplicate node is added, ref_info should not be mutated.

    This is a different bug from #23788 (which is about mutation on failure).
    We only verify that our cleanup logic doesn't run when allow_update=False.
    """
    store = SimpleDocumentStore()

    node = TextNode(
        text="Original",
        id_="dup_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_A")},
    )
    store.add_documents([node])

    # Try to add same node again with allow_update=False - should raise
    node_dup = TextNode(
        text="Duplicate",
        id_="dup_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    with pytest.raises(ValueError, match="already exists"):
        store.add_documents([node_dup], allow_update=False)

    # ref_A should still have dup_node (no mutation from failed add)
    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is not None
    assert "dup_node" in ref_a_info.node_ids, (
        "ref_A should still have dup_node after failed add"
    )


@pytest.mark.parametrize(
    ("initial_ref", "final_ref"),
    [
        ("ref_A", "ref_A"),
        ("ref_A", "ref_C"),
        ("ref_A", None),
        (None, "ref_A"),
        (None, "ref_C"),
        (None, None),
    ],
)
def test_batch_same_node_b_to_final_param(
    initial_ref: str | None, final_ref: str | None
) -> None:
    """
    Parameterized: batch [ref_B, final_ref] for same node.
    Covers: existing A -> [B,A], [B,C], [B,None], new node -> [B,A], [B,C], [B,None].
    """
    store = SimpleDocumentStore()

    node_id = "param_node"

    # Optional initial state
    if initial_ref is not None:
        node_init = TextNode(
            text="Init",
            id_=node_id,
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=initial_ref)
            },
        )
        store.add_documents([node_init])

    # Batch: ref_B then final_ref
    node_b = TextNode(
        text="Batch B",
        id_=node_id,
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    if final_ref is not None:
        node_final = TextNode(
            text="Batch final",
            id_=node_id,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=final_ref)},
        )
    else:
        node_final = TextNode(text="Batch final", id_=node_id)

    store.add_documents([node_b, node_final], allow_update=True)

    # Assert final ref has node exactly once (if final_ref exists)
    if final_ref is not None:
        final_info = store.get_ref_doc_info(final_ref)
        assert final_info is not None
        assert final_info.node_ids.count(node_id) == 1, (
            f"{final_ref} should have {node_id} exactly once"
        )

    # Assert intermediate ref_B does NOT have node
    ref_b_info = store.get_ref_doc_info("ref_B")
    if ref_b_info is not None:
        assert node_id not in ref_b_info.node_ids, (
            f"ref_B should not have {node_id} (intermediate)"
        )

    # Assert old initial_ref does NOT have node (if different from final)
    if initial_ref is not None and initial_ref != final_ref:
        init_info = store.get_ref_doc_info(initial_ref)
        if init_info is not None:
            assert node_id not in init_info.node_ids, (
                f"{initial_ref} should not have {node_id} (old stored ref)"
            )

    # Node itself should exist
    assert store.document_exists(node_id), f"{node_id} should exist"

    # Verify node's ref_doc_id matches final
    final_node = store.get_node(node_id)
    assert final_node.ref_doc_id == final_ref, f"Final ref_doc_id should be {final_ref}"

    # Verify final content (last write wins)
    assert final_node.get_content() == "Batch final", (
        f"Final content should be 'Batch final'"
    )

    # Delete non-final refs and verify node survives
    non_final_refs = set()
    if initial_ref is not None and initial_ref != final_ref:
        non_final_refs.add(initial_ref)
    non_final_refs.add("ref_B")

    for ref in non_final_refs:
        store.delete_ref_doc(ref, raise_error=False)

    # Node should still exist after deleting non-final refs
    assert store.document_exists(node_id), (
        f"{node_id} should survive deleting non-final refs"
    )

    # Verify final node state after cleanup
    final_node_after = store.get_node(node_id)
    assert final_node_after.ref_doc_id == final_ref
    assert final_node_after.get_content() == "Batch final"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_ref", "final_ref"),
    [
        ("ref_A", "ref_A"),
        ("ref_A", "ref_C"),
        ("ref_A", None),
        (None, "ref_A"),
        (None, "ref_C"),
        (None, None),
    ],
)
async def test_async_batch_same_node_b_to_final_param(
    initial_ref: str | None, final_ref: str | None
) -> None:
    """Async parameterized version."""
    store = SimpleDocumentStore()
    node_id = "async_param_node"

    if initial_ref is not None:
        node_init = TextNode(
            text="Init",
            id_=node_id,
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=initial_ref)
            },
        )
        await store.async_add_documents([node_init])

    node_b = TextNode(
        text="Batch B",
        id_=node_id,
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    if final_ref is not None:
        node_final = TextNode(
            text="Batch final",
            id_=node_id,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=final_ref)},
        )
    else:
        node_final = TextNode(text="Batch final", id_=node_id)

    await store.async_add_documents([node_b, node_final], allow_update=True)

    if final_ref is not None:
        final_info = store.get_ref_doc_info(final_ref)
        assert final_info is not None
        assert final_info.node_ids.count(node_id) == 1

    ref_b_info = store.get_ref_doc_info("ref_B")
    if ref_b_info is not None:
        assert node_id not in ref_b_info.node_ids

    if initial_ref is not None and initial_ref != final_ref:
        init_info = store.get_ref_doc_info(initial_ref)
        if init_info is not None:
            assert node_id not in init_info.node_ids

    assert store.document_exists(node_id)

    final_node = store.get_node(node_id)
    assert final_node.ref_doc_id == final_ref
    assert final_node.get_content() == "Batch final"

    # Delete non-final refs and verify survival
    non_final_refs = set()
    if initial_ref is not None and initial_ref != final_ref:
        non_final_refs.add(initial_ref)
    non_final_refs.add("ref_B")

    for ref in non_final_refs:
        await store.adelete_ref_doc(ref, raise_error=False)

    assert store.document_exists(node_id)

    final_node_after = store.get_node(node_id)
    assert final_node_after.ref_doc_id == final_ref
    assert final_node_after.get_content() == "Batch final"


def test_batch_same_node_shared_member_at_intermediate_ref() -> None:
    """
    Shared member at ref_B: batch [move_node(B), move_node(C)] with other_node at B.
    After batch: move_node ends at C, other_node stays at B.
    Delete B: other_node deleted, move_node survives.
    """
    store = SimpleDocumentStore()

    # Pre-populate ref_B with two nodes
    node_move = TextNode(
        text="Move",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    node_stay = TextNode(
        text="Stay at B",
        id_="stay_node_b",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    store.add_documents([node_move, node_stay])

    # Batch: move_node goes B -> C
    node_b = TextNode(
        text="Move batch B",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_B")},
    )
    node_c = TextNode(
        text="Move batch C",
        id_="move_node",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ref_C")},
    )
    store.add_documents([node_b, node_c], allow_update=True)

    # ref_B should have stay_node_b but not move_node
    ref_b_info = store.get_ref_doc_info("ref_B")
    assert ref_b_info is not None
    assert "stay_node_b" in ref_b_info.node_ids, "stay_node_b should remain in ref_B"
    assert "move_node" not in ref_b_info.node_ids, "move_node should be gone from ref_B"

    # ref_C should have move_node
    ref_c_info = store.get_ref_doc_info("ref_C")
    assert ref_c_info is not None
    assert "move_node" in ref_c_info.node_ids

    # Delete ref_B: stay_node_b deleted, move_node survives
    store.delete_ref_doc("ref_B")
    assert not store.document_exists("stay_node_b"), "stay_node_b should be deleted"
    assert store.document_exists("move_node"), "move_node should survive"


# === Fault injection tests: metadata must not be mutated when write fails ===


class FaultInjectedKVStore(SimpleKVStore):
    """
    KVStore that raises on every put_all/aput_all while armed.

    Starts disarmed; call arm() to make subsequent writes raise immediately
    without calling super(). This tests that prepare logic does not mutate
    stored metadata before the write actually succeeds.
    """

    def __init__(self) -> None:
        super().__init__()
        self._armed = False

    def arm(self) -> None:
        self._armed = True

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = "data",
        batch_size: int = 1,
    ) -> None:
        if self._armed:
            raise RuntimeError("Injected write failure")
        super().put_all(kv_pairs, collection, batch_size=batch_size)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = "data",
        batch_size: int = 1,
    ) -> None:
        if self._armed:
            raise RuntimeError("Injected write failure")
        await super().aput_all(kv_pairs, collection, batch_size=batch_size)


def _make_node(node_id: str, ref_id: str) -> TextNode:
    return TextNode(
        text="Node content",
        id_=node_id,
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref_id)},
    )


def test_reparent_failed_write_metadata_unchanged() -> None:
    """
    When put_all raises before first write, stored metadata must not change.

    Regression: stale_ref_info.node_ids.remove() was mutating in-memory list
    before the write, so a failed write left metadata corrupted.
    """
    kv = FaultInjectedKVStore()
    store = KVDocumentStore(kvstore=kv)
    store.add_documents([_make_node("n1", "ref_A")], allow_update=True)

    # Snapshot raw store state before failed write
    raw_before = json.dumps(kv.to_dict(), sort_keys=True)

    # Arm fault - next write should fail immediately
    kv.arm()

    # Now try to reparent to ref_B - this should raise
    with pytest.raises(RuntimeError, match="Injected write failure"):
        store.add_documents([_make_node("n1", "ref_B")], allow_update=True)

    # Raw store state must be completely unchanged
    raw_after = json.dumps(kv.to_dict(), sort_keys=True)
    assert raw_before == raw_after, (
        "Raw store data must be unchanged after failed write"
    )

    # Metadata must be unchanged - node still owned by ref_A
    ref_a_info = store.get_ref_doc_info("ref_A")
    assert ref_a_info is not None
    assert "n1" in ref_a_info.node_ids, "node must still be in ref_A after failed write"
    node_after = store.get_node("n1")
    assert node_after.ref_doc_id == "ref_A"


@pytest.mark.asyncio
async def test_async_reparent_failed_write_metadata_unchanged() -> None:
    """Async: when aput_all raises before write, stored metadata must not change."""
    kv = FaultInjectedKVStore()
    store = KVDocumentStore(kvstore=kv)
    await store.async_add_documents([_make_node("n1", "ref_A")], allow_update=True)

    raw_before = json.dumps(kv.to_dict(), sort_keys=True)

    kv.arm()
    with pytest.raises(RuntimeError, match="Injected write failure"):
        await store.async_add_documents([_make_node("n1", "ref_B")], allow_update=True)

    raw_after = json.dumps(kv.to_dict(), sort_keys=True)
    assert raw_before == raw_after, (
        "Raw store data must be unchanged after failed async write"
    )

    ref_a_info = await store.aget_ref_doc_info("ref_A")
    assert ref_a_info is not None
    assert "n1" in ref_a_info.node_ids
    node_after = await store.aget_node("n1")
    assert node_after.ref_doc_id == "ref_A"

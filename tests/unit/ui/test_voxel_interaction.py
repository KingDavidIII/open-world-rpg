"""Editable voxel resolution, interaction, hotbar, physics, and ray tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from open_world_rpg.gameplay import ItemType, PlayerInventory
from open_world_rpg.ui.voxel import (
    BlockColumn,
    BlockType,
    EditableVoxelWorld,
    InteractionOutcome,
    InteractionResult,
    PlayerState,
    RayHit,
    VoxelHotbar,
    VoxelInteractionController,
    invalidated_chunks_for_edit,
    move_player,
    player_intersects_block,
    ray_cast,
)
from open_world_rpg.world import (
    BlockEditStore,
    BlockMaterial,
    ChunkCoordinate,
    WorldBlockCoordinate,
)


def column_at(_x: int, _z: int) -> BlockColumn:
    return BlockColumn(
        ground_height=4,
        surface_height=7,
        surface=BlockType.GRASS,
        subsurface=BlockType.DIRT,
        water=BlockType.WATER,
    )


def make_world() -> tuple[EditableVoxelWorld, BlockEditStore]:
    edits = BlockEditStore()
    return EditableVoxelWorld(column_at=column_at, edits=edits), edits


DEFAULT_HIT_COORDINATE = WorldBlockCoordinate(x=0, y=4, z=0)


def hit(
    *,
    coordinate: WorldBlockCoordinate = DEFAULT_HIT_COORDINATE,
    material: BlockMaterial = BlockMaterial.GRASS,
    normal: tuple[int, int, int] = (0, 1, 0),
) -> RayHit:
    return RayHit(
        x=coordinate.x,
        y=coordinate.y,
        z=coordinate.z,
        distance=1.0,
        material=material,
        face_normal=normal,
    )


def test_editable_world_resolution_and_override_precedence() -> None:
    world, edits = make_world()
    assert world.edits is edits
    assert world.material_at(0, 4, 0) is BlockMaterial.GRASS
    assert world.material_at(0, 3, 0) is BlockMaterial.DIRT
    assert world.material_at(0, 0, 0) is BlockMaterial.STONE
    assert world.material_at(0, 5, 0) is BlockMaterial.WATER
    assert world.material_at(0, 7, 0) is BlockMaterial.AIR
    coordinate = WorldBlockCoordinate(x=0, y=4, z=0)
    edits.set_block(coordinate, BlockMaterial.SNOW)
    assert world.block_at(coordinate) is BlockMaterial.SNOW
    assert world.solid_at(0, 4, 0)
    edits.set_block(coordinate, BlockMaterial.AIR)
    assert not world.solid_at(0, 4, 0)
    assert world.supports(WorldBlockCoordinate(x=0, y=0, z=0))
    assert world.supports(WorldBlockCoordinate(x=0, y=64, z=0))
    assert not world.supports(WorldBlockCoordinate(x=0, y=-1, z=0))
    with pytest.raises(TypeError):
        EditableVoxelWorld(column_at=cast(Any, None), edits=edits)
    with pytest.raises(TypeError):
        EditableVoxelWorld(column_at=column_at, edits=cast(Any, object()))
    with pytest.raises(TypeError):
        world.block_at(cast(Any, (0, 0, 0)))
    with pytest.raises(TypeError):
        world.supports(cast(Any, (0, 0, 0)))


def test_hotbar_selection_validation_and_wheel_wrapping() -> None:
    hotbar = VoxelHotbar()
    assert hotbar.selected_material is BlockMaterial.GRASS
    assert hotbar.select(5).selected_material is BlockMaterial.SNOW
    assert hotbar.select(9).selected_material is None
    assert hotbar.cycle(1).selected_index == 8
    assert hotbar.cycle(-1).selected_index == 1
    with pytest.raises(ValueError):
        VoxelHotbar(slots=(None,))
    with pytest.raises(TypeError):
        VoxelHotbar(slots=cast(Any, (object(),) * 9))
    with pytest.raises(TypeError):
        VoxelHotbar(selected_index=True)
    with pytest.raises(ValueError):
        VoxelHotbar(selected_index=9)
    with pytest.raises(TypeError):
        hotbar.select(True)
    with pytest.raises(ValueError):
        hotbar.select(0)
    with pytest.raises(TypeError):
        hotbar.cycle(True)


def test_breaking_precedence_cooldown_and_water_policy() -> None:
    world, edits = make_world()
    controller = VoxelInteractionController(world=world, edits=edits)
    assert controller.break_block(target=None, now=0).result is InteractionResult.NO_TARGET
    assert (
        controller.break_block(target=hit(material=BlockMaterial.WATER), now=0).result
        is InteractionResult.WATER
    )
    assert (
        controller.break_block(target=hit(material=BlockMaterial.AIR), now=0).result
        is InteractionResult.NO_TARGET
    )
    outcome = controller.break_block(target=hit(), now=0)
    assert outcome.changed
    assert edits.get(hit().coordinate).material is BlockMaterial.AIR  # type: ignore[union-attr]
    assert controller.break_block(target=hit(), now=0.1).result is InteractionResult.COOLDOWN


def test_break_reports_drop_and_inventory_placement_consumes_atomically() -> None:
    world, edits = make_world()
    controller = VoxelInteractionController(
        world=world,
        edits=edits,
        break_cooldown=0,
        placement_cooldown=0,
    )
    broken = controller.break_block(target=hit(material=BlockMaterial.GRASS), now=0)
    assert broken.dropped_item is ItemType.GRASS_BLOCK
    inventory = PlayerInventory()
    inventory.add(ItemType.STONE_BLOCK, 2)
    placed = controller.place_inventory_block(
        target=hit(coordinate=WorldBlockCoordinate(x=0, y=7, z=0)),
        inventory=inventory,
        player=PlayerState(x=4, y=4, z=4),
        now=0,
    )
    assert placed.result is InteractionResult.PLACED
    assert inventory.selected_stack is not None
    assert inventory.selected_stack.quantity == 1
    before = inventory.snapshot()
    assert (
        controller.place_inventory_block(
            target=None,
            inventory=inventory,
            player=PlayerState(x=4, y=4, z=4),
            now=1,
        ).result
        is InteractionResult.NO_TARGET
    )
    assert inventory.snapshot() == before
    inventory.clear()
    assert (
        controller.place_inventory_block(
            target=hit(coordinate=WorldBlockCoordinate(x=0, y=9, z=0)),
            inventory=inventory,
            player=PlayerState(x=4, y=4, z=4),
            now=2,
        ).result
        is InteractionResult.EMPTY_SLOT
    )
    with pytest.raises(TypeError):
        controller.place_inventory_block(
            target=hit(),
            inventory="bad",  # type: ignore[arg-type]
            player=PlayerState(x=4, y=4, z=4),
            now=3,
        )


def test_placement_validation_and_success() -> None:
    world, edits = make_world()
    controller = VoxelInteractionController(world=world, edits=edits)
    player = PlayerState(x=10.5, y=10.0, z=10.5)
    assert (
        controller.place_block(
            target=None, material=BlockMaterial.STONE, player=player, now=0
        ).result
        is InteractionResult.NO_TARGET
    )
    assert (
        controller.place_block(target=hit(), material=None, player=player, now=0).result
        is InteractionResult.EMPTY_SLOT
    )
    assert (
        controller.place_block(
            target=hit(), material=BlockMaterial.WATER, player=player, now=0
        ).result
        is InteractionResult.EMPTY_SLOT
    )
    occupied = hit(normal=(0, -1, 0))
    assert (
        controller.place_block(
            target=occupied, material=BlockMaterial.STONE, player=player, now=0
        ).result
        is InteractionResult.OCCUPIED
    )
    outside = hit(
        coordinate=WorldBlockCoordinate(x=0, y=64, z=0),
        normal=(0, 1, 0),
    )
    assert (
        controller.place_block(
            target=outside, material=BlockMaterial.STONE, player=player, now=0
        ).result
        is InteractionResult.OUT_OF_BOUNDS
    )
    player_target = hit(
        coordinate=WorldBlockCoordinate(x=10, y=9, z=10),
        normal=(0, 1, 0),
    )
    assert (
        controller.place_block(
            target=player_target,
            material=BlockMaterial.STONE,
            player=player,
            now=0,
        ).result
        is InteractionResult.PLAYER_INTERSECTION
    )
    outcome = controller.place_block(
        target=hit(coordinate=WorldBlockCoordinate(x=0, y=7, z=0)),
        material=BlockMaterial.STONE,
        player=player,
        now=0,
    )
    assert outcome.result is InteractionResult.PLACED
    assert edits.get(WorldBlockCoordinate(x=0, y=8, z=0)).material is BlockMaterial.STONE  # type: ignore[union-attr]
    assert (
        controller.place_block(
            target=hit(), material=BlockMaterial.STONE, player=player, now=0.1
        ).result
        is InteractionResult.COOLDOWN
    )


def test_boundary_mesh_invalidation_is_owner_and_neighbour_only() -> None:
    corner = WorldBlockCoordinate(x=-16, y=3, z=15)
    assert invalidated_chunks_for_edit(corner) == (
        ChunkCoordinate(x=-2, y=0),
        ChunkCoordinate(x=-1, y=0),
        ChunkCoordinate(x=-1, y=1),
    )
    assert invalidated_chunks_for_edit(WorldBlockCoordinate(x=2, y=3, z=2)) == (
        ChunkCoordinate(x=0, y=0),
    )
    assert invalidated_chunks_for_edit(WorldBlockCoordinate(x=15, y=3, z=2)) == (
        ChunkCoordinate(x=0, y=0),
        ChunkCoordinate(x=1, y=0),
    )
    with pytest.raises(TypeError):
        invalidated_chunks_for_edit(cast(Any, (0, 0, 0)))


def test_raycast_returns_material_normal_and_negative_adjacent_coordinate() -> None:
    result = ray_cast(
        origin=(-0.2, 2.5, 0.5),
        direction=(0.0, -1.0, 0.0),
        block_at=lambda _x, y, _z: BlockMaterial.STONE if y == 0 else BlockMaterial.AIR,
    )
    assert result is not None
    assert result.coordinate == WorldBlockCoordinate(x=-1, y=0, z=0)
    assert result.material is BlockMaterial.STONE
    assert result.face_normal == (0, 1, 0)
    assert result.adjacent_coordinate == WorldBlockCoordinate(x=-1, y=1, z=0)
    with pytest.raises(TypeError):
        ray_cast(origin=(0, 0, 0), direction=(0, 1, 0))


def test_player_intersection_placed_collision_support_and_removed_gravity() -> None:
    block = WorldBlockCoordinate(x=0, y=1, z=0)
    player = PlayerState(x=0.5, y=2.0, z=0.5, grounded=True)
    assert not player_intersects_block(player=player, coordinate=block)
    assert player_intersects_block(
        player=player,
        coordinate=WorldBlockCoordinate(x=0, y=2, z=0),
    )
    with pytest.raises(TypeError):
        player_intersects_block(player=cast(Any, object()), coordinate=block)
    with pytest.raises(TypeError):
        player_intersects_block(player=player, coordinate=cast(Any, (0, 1, 0)))

    solids = {(0, 1, 0)}
    standing = move_player(
        player=player,
        delta_x=0,
        delta_z=0,
        delta_seconds=0.05,
        height_at=lambda _x, _z: 0,
        solid_at=lambda x, y, z: (x, y, z) in solids,
    )
    assert standing.grounded
    solids.clear()
    falling = move_player(
        player=standing,
        delta_x=0,
        delta_z=0,
        delta_seconds=0.05,
        height_at=lambda _x, _z: 0,
        solid_at=lambda x, y, z: (x, y, z) in solids,
    )
    assert not falling.grounded
    assert falling.y < standing.y

    wall = {(1, 2, 0), (0, 2, 1)}
    blocked_x = move_player(
        player=player,
        delta_x=1,
        delta_z=0,
        delta_seconds=0,
        height_at=lambda _x, _z: 0,
        solid_at=lambda x, y, z: (x, y, z) in wall,
    )
    assert blocked_x.x == player.x
    blocked_z = move_player(
        player=player,
        delta_x=0,
        delta_z=1,
        delta_seconds=0,
        height_at=lambda _x, _z: 0,
        solid_at=lambda x, y, z: (x, y, z) in wall,
    )
    assert blocked_z.z == player.z
    flying = move_player(
        player=PlayerState(x=0.5, y=2, z=0.5, flying=True),
        delta_x=0,
        delta_z=0,
        delta_seconds=0,
        height_at=lambda _x, _z: 0,
        solid_at=lambda _x, _y, _z: False,
    )
    assert flying.flying
    head_blocked = move_player(
        player=PlayerState(
            x=0.5,
            y=2,
            z=0.5,
            vertical_velocity=5,
            grounded=False,
        ),
        delta_x=0,
        delta_z=0,
        delta_seconds=0.1,
        height_at=lambda _x, _z: 0,
        solid_at=lambda _x, y, _z: y == 4,
    )
    assert head_blocked.y == 2
    assert head_blocked.vertical_velocity == 0


def test_interaction_constructor_validation() -> None:
    world, edits = make_world()
    with pytest.raises(TypeError):
        VoxelInteractionController(world=cast(Any, object()), edits=edits)
    with pytest.raises(TypeError):
        VoxelInteractionController(world=world, edits=cast(Any, object()))
    with pytest.raises(ValueError):
        VoxelInteractionController(world=world, edits=edits, break_cooldown=-1)


def test_validated_inventory_consumption_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, edits = make_world()
    controller = VoxelInteractionController(world=world, edits=edits)
    inventory = PlayerInventory()
    inventory.add(ItemType.STONE_BLOCK, 1)
    monkeypatch.setattr(
        controller,
        "place_block",
        lambda **_kwargs: InteractionOutcome(result=InteractionResult.PLACED),
    )
    monkeypatch.setattr(inventory, "remove_from_slot", lambda *_args: False)
    with pytest.raises(RuntimeError, match="consumption failed"):
        controller.place_inventory_block(
            target=hit(),
            inventory=inventory,
            player=PlayerState(x=4, y=4, z=4),
            now=0,
        )

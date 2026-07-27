"""Tools, deterministic mining, and survival-vitals contracts."""

from dataclasses import FrozenInstanceError

import pytest

from open_world_rpg.gameplay import (
    FALL_DAMAGE_PER_BLOCK,
    InventorySlot,
    ItemStack,
    ItemType,
    MiningStatus,
    PlayerInventory,
    PlayerVitals,
    PlayerVitalsSnapshot,
    TimedMiningController,
    ToolClassification,
    ToolInstance,
    ToolTier,
    hardness_microseconds,
    item_for_material,
    item_policy,
    material_for_item,
    mining_duration_microseconds,
    tool_speed_multiplier,
)
from open_world_rpg.world import BlockMaterial, WorldBlockCoordinate


def test_catalogue_centralises_block_and_tool_policy() -> None:
    block = item_policy(ItemType.STONE_BLOCK)
    assert (block.stackable, block.maximum_stack_size, block.placeable) == (True, 64, True)
    assert block.placeable_material is BlockMaterial.STONE
    assert block.maximum_durability is None
    wooden = item_policy(ItemType.WOODEN_PICKAXE)
    stone = item_policy(ItemType.STONE_SHOVEL)
    assert (wooden.tool_classification, wooden.tool_tier, wooden.maximum_durability) == (
        ToolClassification.PICKAXE,
        ToolTier.WOOD,
        64,
    )
    assert (stone.tool_classification, stone.tool_tier, stone.maximum_durability) == (
        ToolClassification.SHOVEL,
        ToolTier.STONE,
        128,
    )
    assert not wooden.stackable and not wooden.placeable
    with pytest.raises(TypeError):
        item_policy("wooden_pickaxe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        material_for_item(ItemType.WOODEN_PICKAXE)
    assert item_for_material(BlockMaterial.WOOD) is ItemType.WOOD_LOG
    assert item_for_material(BlockMaterial.LEAVES) is None
    with pytest.raises(ValueError):
        ItemStack(item=ItemType.WOODEN_SHOVEL, quantity=1)


def test_tool_instance_is_validated_immutable_and_destroyed_at_zero() -> None:
    tool = ToolInstance.create(ItemType.WOODEN_PICKAXE)
    assert tool.current_durability == tool.maximum_durability == 64
    used = tool.use()
    assert used == ToolInstance(
        item=ItemType.WOODEN_PICKAXE,
        current_durability=63,
        maximum_durability=64,
    )
    assert tool.current_durability == 64
    assert (
        ToolInstance(
            item=ItemType.WOODEN_PICKAXE,
            current_durability=1,
            maximum_durability=64,
        ).use()
        is None
    )
    with pytest.raises(FrozenInstanceError):
        tool.current_durability = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        ToolInstance.create(ItemType.DIRT_BLOCK)
    with pytest.raises(ValueError):
        ToolInstance(item=ItemType.DIRT_BLOCK, current_durability=1, maximum_durability=64)
    for value in (True, 1.5, "1"):
        with pytest.raises(TypeError):
            ToolInstance(
                item=ItemType.WOODEN_PICKAXE,
                current_durability=value,  # type: ignore[arg-type]
                maximum_durability=64,
            )
    for value in (0, -1):
        with pytest.raises(ValueError):
            ToolInstance(
                item=ItemType.WOODEN_PICKAXE,
                current_durability=value,
                maximum_durability=64,
            )
    with pytest.raises(ValueError):
        ToolInstance(
            item=ItemType.WOODEN_PICKAXE,
            current_durability=65,
            maximum_durability=64,
        )
    with pytest.raises(ValueError):
        ToolInstance(
            item=ItemType.WOODEN_PICKAXE,
            current_durability=1,
            maximum_durability=128,
        )


def test_mixed_inventory_tool_mutations_are_atomic() -> None:
    inventory = PlayerInventory()
    tool = ToolInstance.create(ItemType.WOODEN_PICKAXE)
    assert inventory.add_tool(tool, slot=0)
    assert inventory.selected_tool is tool and inventory.selected_stack is None
    before = inventory.snapshot()
    assert not inventory.add_tool(tool, slot=0)
    assert inventory.snapshot() == before
    assert inventory.use_tool(0)
    assert inventory.revision == before.revision + 1
    assert inventory.selected_tool is not None
    assert inventory.selected_tool.current_durability == 63
    assert not inventory.use_tool(1)
    replacement = ToolInstance(
        item=ItemType.WOODEN_PICKAXE,
        current_durability=1,
        maximum_durability=64,
    )
    inventory.set_slot(0, replacement)
    assert inventory.use_tool(0)
    assert inventory.slot(0) is None
    assert inventory.add_tool(tool)
    assert inventory.remove_tool(0) == tool
    assert inventory.remove_tool(0) is None
    with pytest.raises(TypeError):
        inventory.add_tool("tool")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        inventory.add(ItemType.WOODEN_PICKAXE, 1)
    with pytest.raises(ValueError):
        inventory.remove(ItemType.WOODEN_PICKAXE, 1)
    assert inventory.total_quantity(ItemType.WOODEN_PICKAXE) == 0
    _: InventorySlot = tool
    full = PlayerInventory()
    for index in range(27):
        full.set_slot(index, ItemStack(item=ItemType.DIRT_BLOCK, quantity=1))
    snapshot = full.snapshot()
    assert not full.add_tool(tool)
    assert full.snapshot() == snapshot


def test_mining_policy_known_hardness_and_effectiveness() -> None:
    assert {
        material: hardness_microseconds(material)
        for material in (
            BlockMaterial.SNOW,
            BlockMaterial.SAND,
            BlockMaterial.DIRT,
            BlockMaterial.GRASS,
            BlockMaterial.WOOD,
            BlockMaterial.LEAVES,
            BlockMaterial.STONE,
        )
    } == {
        BlockMaterial.SNOW: 300_000,
        BlockMaterial.SAND: 450_000,
        BlockMaterial.DIRT: 550_000,
        BlockMaterial.GRASS: 650_000,
        BlockMaterial.WOOD: 900_000,
        BlockMaterial.LEAVES: 180_000,
        BlockMaterial.STONE: 2_000_000,
    }
    assert hardness_microseconds(BlockMaterial.AIR) is None
    assert hardness_microseconds(BlockMaterial.WATER) is None
    with pytest.raises(TypeError):
        hardness_microseconds("stone")  # type: ignore[arg-type]
    wood_pick = ToolInstance.create(ItemType.WOODEN_PICKAXE)
    stone_pick = ToolInstance.create(ItemType.STONE_PICKAXE)
    wood_shovel = ToolInstance.create(ItemType.WOODEN_SHOVEL)
    stone_shovel = ToolInstance.create(ItemType.STONE_SHOVEL)
    assert tool_speed_multiplier(None, BlockMaterial.STONE) == 100
    assert tool_speed_multiplier(wood_pick, BlockMaterial.STONE) == 225
    assert tool_speed_multiplier(stone_pick, BlockMaterial.STONE) == 375
    assert tool_speed_multiplier(wood_shovel, BlockMaterial.STONE) == 75
    assert tool_speed_multiplier(stone_shovel, BlockMaterial.STONE) == 85
    assert tool_speed_multiplier(wood_shovel, BlockMaterial.DIRT) == 225
    assert tool_speed_multiplier(stone_shovel, BlockMaterial.SNOW) == 375
    assert tool_speed_multiplier(None, BlockMaterial.WATER) == 0
    assert mining_duration_microseconds(BlockMaterial.STONE, wood_pick) == 888_889


def test_timed_mining_controller_cancels_and_completes_exactly_once() -> None:
    controller = TimedMiningController()
    coordinate = WorldBlockCoordinate(x=-1, y=3, z=2)
    tool = ToolInstance.create(ItemType.WOODEN_PICKAXE)
    assert controller.snapshot.status is MiningStatus.IDLE
    assert controller.snapshot.normalised_progress == 0.0
    assert controller.begin(target=coordinate, material=BlockMaterial.STONE, tool=tool)
    assert not controller.begin(target=coordinate, material=BlockMaterial.STONE, tool=tool)
    assert controller.advance(444_444).normalised_progress < 0.5
    completed = controller.advance(999_999)
    assert completed.status is MiningStatus.COMPLETED
    assert completed.normalised_progress == 1.0
    assert controller.advance(1) == completed
    assert controller.cancel("released")
    assert controller.snapshot.status is MiningStatus.CANCELLED
    assert controller.snapshot.last_cancellation_reason == "released"
    assert not controller.cancel("released")
    assert controller.reset()
    assert not controller.reset()
    assert not controller.begin(target=coordinate, material=BlockMaterial.WATER, tool=None)
    with pytest.raises(TypeError):
        controller.begin(target="bad", material=BlockMaterial.STONE, tool=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        controller.begin(
            target=coordinate,
            material=BlockMaterial.STONE,
            tool="bad",  # type: ignore[arg-type]
        )
    for value in (True, 1.5):
        with pytest.raises(TypeError):
            controller.advance(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        controller.advance(-1)
    with pytest.raises(TypeError):
        controller.cancel(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        controller.cancel(" ")


def test_vitals_stamina_falls_damage_and_respawn() -> None:
    vitals = PlayerVitals()
    assert vitals.can_sprint
    assert vitals.update_stamina(1_000_000, sprinting=True)
    assert vitals.snapshot.stamina == 82
    assert vitals.update_stamina(1_000_000, sprinting=False)
    assert vitals.snapshot.stamina == 82
    assert vitals.update_stamina(1_000_000, sprinting=False)
    assert vitals.snapshot.stamina == 96
    assert vitals.jump()
    assert vitals.snapshot.stamina == 84
    assert vitals.record_airborne_descent(2_999)
    assert vitals.land() == 0
    vitals.record_airborne_descent(5_000)
    assert vitals.land() == 2 * FALL_DAMAGE_PER_BLOCK
    assert vitals.snapshot.health == 84
    vitals.record_airborne_descent(20_000)
    assert vitals.land(in_water=True) == 0
    vitals.record_airborne_descent(20_000)
    assert vitals.land(immune=True) == 0
    assert vitals.damage(84)
    assert vitals.snapshot.health == 0
    assert not vitals.damage(1)
    assert vitals.respawn()
    assert (vitals.snapshot.health, vitals.snapshot.stamina, vitals.snapshot.death_count) == (
        100,
        100,
        1,
    )
    assert vitals.reset_fall() is False


def test_vitals_validation_restore_clamps_and_rejections() -> None:
    vitals = PlayerVitals()
    original = vitals.snapshot
    assert not vitals.update_stamina(0, sprinting=False)
    assert not vitals.update_stamina(1, sprinting=False, active=False)
    for value in (True, 1.5):
        with pytest.raises(TypeError):
            vitals.update_stamina(value, sprinting=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        vitals.update_stamina(-1, sprinting=False)
    with pytest.raises(TypeError):
        vitals.update_stamina(1, sprinting=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vitals.land(in_water=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vitals.restore("bad")  # type: ignore[arg-type]
    assert not vitals.restore(original)
    exhausted = PlayerVitalsSnapshot(stamina_milli=4_999)
    assert vitals.restore(exhausted)
    assert not vitals.jump()
    assert not vitals.can_sprint
    with pytest.raises(TypeError):
        PlayerVitalsSnapshot(revision=True)
    with pytest.raises(TypeError):
        PlayerVitalsSnapshot(grounded=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlayerVitalsSnapshot(maximum_health_milli=0)
    with pytest.raises(ValueError):
        PlayerVitalsSnapshot(maximum_stamina_milli=0)
    with pytest.raises(ValueError):
        PlayerVitalsSnapshot(health_milli=101_000)
    with pytest.raises(ValueError):
        PlayerVitalsSnapshot(stamina_milli=101_000)

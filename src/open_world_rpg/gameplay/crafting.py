"""Deterministic recipe catalogue and atomic inventory crafting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .inventory import PlayerInventory, PlayerInventorySnapshot
from .items import ItemType, ToolInstance, item_policy


class CraftingResult(StrEnum):
    """Stable outcome identifiers for crafting attempts."""

    CRAFTED = "crafted"
    UNKNOWN_RECIPE = "unknown recipe"
    MISSING_INGREDIENTS = "missing ingredients"
    INVENTORY_FULL = "inventory full"


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RecipeIngredient:
    """One positive stackable ingredient requirement."""

    item: ItemType
    quantity: int

    def __post_init__(self) -> None:
        if not item_policy(self.item).stackable:
            raise ValueError("recipe ingredients must be stackable items.")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer.")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CraftingRecipe:
    """One catalogue recipe with a stack or tool output."""

    identifier: str
    display_name: str
    ingredients: tuple[RecipeIngredient, ...]
    output_item: ItemType
    output_quantity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str):
            raise TypeError("identifier must be a string.")
        if not self.identifier or self.identifier.strip() != self.identifier:
            raise ValueError("identifier must be a non-empty canonical string.")
        if not isinstance(self.display_name, str):
            raise TypeError("display_name must be a string.")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty.")
        if not isinstance(self.ingredients, tuple):
            raise TypeError("ingredients must be a tuple.")
        if not self.ingredients:
            raise ValueError("recipes require at least one ingredient.")
        if len({ingredient.item for ingredient in self.ingredients}) != len(self.ingredients):
            raise ValueError("recipe ingredients must not repeat item identities.")
        if isinstance(self.output_quantity, bool) or not isinstance(self.output_quantity, int):
            raise TypeError("output_quantity must be an integer.")
        if self.output_quantity <= 0:
            raise ValueError("output_quantity must be positive.")
        policy = item_policy(self.output_item)
        if not policy.stackable and self.output_quantity != 1:
            raise ValueError("tool recipes must produce exactly one tool.")
        if policy.stackable and self.output_quantity > policy.maximum_stack_size:
            raise ValueError("output_quantity exceeds the output stack maximum.")

    @property
    def output_label(self) -> str:
        suffix = "" if self.output_quantity == 1 else f" x{self.output_quantity}"
        return f"{self.output_item.display_name}{suffix}"


@dataclass(frozen=True, slots=True, kw_only=True)
class CraftingAttempt:
    """Observable result of one atomic crafting request."""

    result: CraftingResult
    recipe: CraftingRecipe | None = None

    @property
    def crafted(self) -> bool:
        return self.result is CraftingResult.CRAFTED


DEFAULT_RECIPES: Final[tuple[CraftingRecipe, ...]] = (
    CraftingRecipe(
        identifier="wood_planks",
        display_name="Wood Planks",
        ingredients=(RecipeIngredient(item=ItemType.WOOD_LOG, quantity=1),),
        output_item=ItemType.WOOD_PLANK,
        output_quantity=4,
    ),
    CraftingRecipe(
        identifier="sticks",
        display_name="Sticks",
        ingredients=(RecipeIngredient(item=ItemType.WOOD_PLANK, quantity=2),),
        output_item=ItemType.STICK,
        output_quantity=4,
    ),
    CraftingRecipe(
        identifier="wooden_pickaxe",
        display_name="Wooden Pickaxe",
        ingredients=(
            RecipeIngredient(item=ItemType.WOOD_PLANK, quantity=3),
            RecipeIngredient(item=ItemType.STICK, quantity=2),
        ),
        output_item=ItemType.WOODEN_PICKAXE,
    ),
    CraftingRecipe(
        identifier="wooden_shovel",
        display_name="Wooden Shovel",
        ingredients=(
            RecipeIngredient(item=ItemType.WOOD_PLANK, quantity=1),
            RecipeIngredient(item=ItemType.STICK, quantity=2),
        ),
        output_item=ItemType.WOODEN_SHOVEL,
    ),
    CraftingRecipe(
        identifier="stone_pickaxe",
        display_name="Stone Pickaxe",
        ingredients=(
            RecipeIngredient(item=ItemType.STONE_BLOCK, quantity=3),
            RecipeIngredient(item=ItemType.STICK, quantity=2),
        ),
        output_item=ItemType.STONE_PICKAXE,
    ),
    CraftingRecipe(
        identifier="stone_shovel",
        display_name="Stone Shovel",
        ingredients=(
            RecipeIngredient(item=ItemType.STONE_BLOCK, quantity=1),
            RecipeIngredient(item=ItemType.STICK, quantity=2),
        ),
        output_item=ItemType.STONE_SHOVEL,
    ),
)


class CraftingCatalogue:
    """Immutable deterministic recipe lookup."""

    def __init__(self, recipes: tuple[CraftingRecipe, ...] = DEFAULT_RECIPES) -> None:
        if not isinstance(recipes, tuple):
            raise TypeError("recipes must be a tuple.")
        if not recipes:
            raise ValueError("catalogue must contain at least one recipe.")
        if any(not isinstance(recipe, CraftingRecipe) for recipe in recipes):
            raise TypeError("recipes must contain CraftingRecipe values.")
        identifiers = tuple(recipe.identifier for recipe in recipes)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("recipe identifiers must be unique.")
        self._recipes = recipes
        self._by_identifier = {recipe.identifier: recipe for recipe in recipes}

    @property
    def recipes(self) -> tuple[CraftingRecipe, ...]:
        return self._recipes

    def recipe(self, identifier: str) -> CraftingRecipe | None:
        if not isinstance(identifier, str):
            raise TypeError("identifier must be a string.")
        return self._by_identifier.get(identifier)


class CraftingService:
    """Apply catalogue recipes to an inventory as one atomic mutation."""

    def __init__(self, catalogue: CraftingCatalogue | None = None) -> None:
        self.catalogue = CraftingCatalogue() if catalogue is None else catalogue
        if not isinstance(self.catalogue, CraftingCatalogue):
            raise TypeError("catalogue must be a CraftingCatalogue.")

    def can_craft(self, inventory: PlayerInventory, recipe: CraftingRecipe) -> bool:
        self._validate(inventory, recipe)
        return all(
            inventory.total_quantity(ingredient.item) >= ingredient.quantity
            for ingredient in recipe.ingredients
        )

    def craft(self, inventory: PlayerInventory, identifier: str) -> CraftingAttempt:
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        recipe = self.catalogue.recipe(identifier)
        if recipe is None:
            return CraftingAttempt(result=CraftingResult.UNKNOWN_RECIPE)
        if not self.can_craft(inventory, recipe):
            return CraftingAttempt(result=CraftingResult.MISSING_INGREDIENTS, recipe=recipe)

        original = inventory.snapshot()
        working = PlayerInventory.from_snapshot(original)
        for ingredient in recipe.ingredients:
            removed = working.remove(ingredient.item, ingredient.quantity)
            assert removed, "validated ingredients must be removable"

        policy = item_policy(recipe.output_item)
        if policy.stackable:
            addition = working.add(recipe.output_item, recipe.output_quantity)
            accepted = addition.remainder == 0
        else:
            accepted = working.add_tool(ToolInstance.create(recipe.output_item))
        if not accepted:
            return CraftingAttempt(result=CraftingResult.INVENTORY_FULL, recipe=recipe)

        inventory.restore(
            PlayerInventorySnapshot(
                revision=original.revision + 1,
                selected_hotbar_index=original.selected_hotbar_index,
                slots=working.slots(),
            )
        )
        return CraftingAttempt(result=CraftingResult.CRAFTED, recipe=recipe)

    @staticmethod
    def _validate(inventory: PlayerInventory, recipe: CraftingRecipe) -> None:
        if not isinstance(inventory, PlayerInventory):
            raise TypeError("inventory must be a PlayerInventory.")
        if not isinstance(recipe, CraftingRecipe):
            raise TypeError("recipe must be a CraftingRecipe.")

"""Renderer-independent voxel menu and screen-state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VoxelScreen(StrEnum):
    MAIN_MENU = "main menu"
    PLAYING = "playing"
    INVENTORY = "inventory"
    PAUSED = "paused"
    DEAD = "dead"


class GameFlowAction(StrEnum):
    NONE = "none"
    NEW_WORLD = "new world"
    CONTINUE = "continue"
    RESUME = "resume"
    SAVE = "save"
    SAVE_AND_QUIT = "save and quit"
    QUIT = "quit"
    RESPAWN = "respawn"


@dataclass(frozen=True, slots=True, kw_only=True)
class MenuOption:
    label: str
    action: GameFlowAction
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("label must be a string.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        if not isinstance(self.action, GameFlowAction):
            raise TypeError("action must be a GameFlowAction.")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")


class GameFlowController:
    """Own menu selection and valid transitions without renderer dependencies."""

    def __init__(
        self,
        *,
        initial_screen: VoxelScreen = VoxelScreen.PLAYING,
        continue_available: bool = False,
    ) -> None:
        if not isinstance(initial_screen, VoxelScreen):
            raise TypeError("initial_screen must be a VoxelScreen.")
        if not isinstance(continue_available, bool):
            raise TypeError("continue_available must be a boolean.")
        self.screen = initial_screen
        self.continue_available = continue_available
        self.selected_index = 0

    @property
    def gameplay_active(self) -> bool:
        return self.screen is VoxelScreen.PLAYING

    @property
    def overlay_active(self) -> bool:
        return self.screen is not VoxelScreen.PLAYING

    @property
    def options(self) -> tuple[MenuOption, ...]:
        if self.screen is VoxelScreen.MAIN_MENU:
            return (
                MenuOption(label="New World", action=GameFlowAction.NEW_WORLD),
                MenuOption(
                    label="Continue",
                    action=GameFlowAction.CONTINUE,
                    enabled=self.continue_available,
                ),
                MenuOption(label="Quit", action=GameFlowAction.QUIT),
            )
        if self.screen is VoxelScreen.PAUSED:
            return (
                MenuOption(label="Resume", action=GameFlowAction.RESUME),
                MenuOption(label="Save", action=GameFlowAction.SAVE),
                MenuOption(label="Save & Quit", action=GameFlowAction.SAVE_AND_QUIT),
                MenuOption(label="Quit Without Saving", action=GameFlowAction.QUIT),
            )
        if self.screen is VoxelScreen.DEAD:
            return (
                MenuOption(label="Respawn", action=GameFlowAction.RESPAWN),
                MenuOption(label="Main Menu", action=GameFlowAction.QUIT),
            )
        return ()

    def set_continue_available(self, available: bool) -> None:
        if not isinstance(available, bool):
            raise TypeError("available must be a boolean.")
        self.continue_available = available
        self._normalise_selection()

    def move_selection(self, direction: int) -> bool:
        if isinstance(direction, bool) or not isinstance(direction, int):
            raise TypeError("direction must be an integer.")
        options = self.options
        if direction == 0 or not options:
            return False
        before = self.selected_index
        step = 1 if direction > 0 else -1
        index = before
        for _ in options:
            index = (index + step) % len(options)
            if options[index].enabled:
                self.selected_index = index
                return index != before
        return False

    def activate_selected(self) -> GameFlowAction:
        options = self.options
        if not options:
            return GameFlowAction.NONE
        self._normalise_selection()
        option = options[self.selected_index]
        return option.action if option.enabled else GameFlowAction.NONE

    def start_new_world(self) -> None:
        self.screen = VoxelScreen.PLAYING
        self.selected_index = 0

    def continue_world(self) -> bool:
        if not self.continue_available:
            return False
        self.screen = VoxelScreen.PLAYING
        self.selected_index = 0
        return True

    def open_inventory(self) -> bool:
        if self.screen is not VoxelScreen.PLAYING:
            return False
        self.screen = VoxelScreen.INVENTORY
        self.selected_index = 0
        return True

    def close_inventory(self) -> bool:
        if self.screen is not VoxelScreen.INVENTORY:
            return False
        self.screen = VoxelScreen.PLAYING
        self.selected_index = 0
        return True

    def pause(self) -> bool:
        if self.screen is not VoxelScreen.PLAYING:
            return False
        self.screen = VoxelScreen.PAUSED
        self.selected_index = 0
        return True

    def resume(self) -> bool:
        if self.screen is not VoxelScreen.PAUSED:
            return False
        self.screen = VoxelScreen.PLAYING
        self.selected_index = 0
        return True

    def mark_dead(self) -> bool:
        if self.screen is VoxelScreen.DEAD:
            return False
        self.screen = VoxelScreen.DEAD
        self.selected_index = 0
        return True

    def respawn(self) -> bool:
        if self.screen is not VoxelScreen.DEAD:
            return False
        self.screen = VoxelScreen.PLAYING
        self.selected_index = 0
        return True

    def return_to_main_menu(self) -> None:
        self.screen = VoxelScreen.MAIN_MENU
        self.selected_index = 0

    def _normalise_selection(self) -> None:
        options = self.options
        if not options:
            self.selected_index = 0
            return
        self.selected_index %= len(options)
        if options[self.selected_index].enabled:
            return
        for index, option in enumerate(options):
            if option.enabled:
                self.selected_index = index
                return

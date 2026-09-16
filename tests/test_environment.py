import pytest

from neural_state_machine import Action, GameConfig, GameObservation, GameSnapshot, ToyGame


def test_reset_returns_literal_initial_observation_and_snapshot() -> None:
    game = ToyGame()

    observation = game.reset()

    assert observation == GameObservation(
        enemy_distance=4 / 6,
        enemy_direction=1,
        health=2 / 3,
        incoming_threat=0.0,
        healing_distance=2 / 6,
        healing_direction=1,
        left_blocked=False,
        right_blocked=False,
    )
    assert game.snapshot() == GameSnapshot(
        player_position=1,
        enemy_position=5,
        player_health=2,
        enemy_health=2,
        healing_available=True,
        done=False,
    )


@pytest.mark.parametrize(
    "config",
    [
        GameConfig(arena_size=2),
        GameConfig(player_start=-1),
        GameConfig(enemy_start=7),
        GameConfig(healing_start=7),
        GameConfig(player_max_health=0),
        GameConfig(player_initial_health=4),
        GameConfig(enemy_initial_health=0),
        GameConfig(attack_damage=0),
    ],
)
def test_game_rejects_invalid_configuration(config: GameConfig) -> None:
    with pytest.raises(ValueError):
        ToyGame(config)


def test_movement_is_clipped_at_both_arena_boundaries() -> None:
    left = ToyGame(GameConfig(player_start=0, enemy_start=5))
    right = ToyGame(GameConfig(player_start=6, enemy_start=1))

    left_result = left.step(Action.MOVE_LEFT)
    right_result = right.step(Action.MOVE_RIGHT)

    assert left_result.snapshot.player_position == 0
    assert right_result.snapshot.player_position == 6


def test_player_cannot_move_into_living_enemy_cell() -> None:
    game = ToyGame(GameConfig(player_start=1, enemy_start=2))

    result = game.step(Action.MOVE_RIGHT)

    assert result.snapshot.player_position == 1
    assert result.snapshot.enemy_position == 2


def test_attack_outside_range_is_penalized_without_damage() -> None:
    result = ToyGame().step(Action.ATTACK)

    assert result.reward == pytest.approx(-0.06)
    assert result.snapshot.enemy_health == 2
    assert result.snapshot.enemy_position == 4


def test_adjacent_attack_hits_then_enemy_retaliates() -> None:
    game = ToyGame(GameConfig(player_start=1, enemy_start=2))

    result = game.step(Action.ATTACK)

    assert result.reward == pytest.approx(-0.01)
    assert result.snapshot.enemy_health == 1
    assert result.snapshot.player_health == 1
    assert not result.done


def test_killing_enemy_ends_game_before_enemy_can_retaliate() -> None:
    game = ToyGame(
        GameConfig(player_start=1, enemy_start=2, enemy_initial_health=1)
    )

    result = game.step(Action.ATTACK)

    assert result.reward == pytest.approx(0.99)
    assert result.snapshot.enemy_health == 0
    assert result.snapshot.player_health == 2
    assert result.done


def test_distant_enemy_approaches_one_cell_after_player_action() -> None:
    result = ToyGame().step(Action.WAIT)

    assert result.snapshot.enemy_position == 4
    assert result.snapshot.player_health == 2


def test_adjacent_enemy_damages_player_without_moving() -> None:
    game = ToyGame(GameConfig(player_start=1, enemy_start=2))

    result = game.step(Action.WAIT)

    assert result.snapshot.enemy_position == 2
    assert result.snapshot.player_health == 1
    assert result.reward == pytest.approx(-0.21)


def test_entering_healing_cell_consumes_item_and_clamps_health() -> None:
    game = ToyGame(GameConfig(player_start=2, enemy_start=6, healing_start=3))

    result = game.step(Action.MOVE_RIGHT)

    assert result.snapshot.player_position == 3
    assert result.snapshot.player_health == 3
    assert not result.snapshot.healing_available
    assert result.reward == pytest.approx(0.09)


def test_enemy_damage_can_kill_player_and_end_game() -> None:
    game = ToyGame(
        GameConfig(player_start=1, enemy_start=2, player_initial_health=1)
    )

    result = game.step(Action.WAIT)

    assert result.snapshot.player_health == 0
    assert result.reward == pytest.approx(-1.21)
    assert result.done

    with pytest.raises(RuntimeError, match="terminal"):
        game.step(Action.WAIT)


def test_step_rejects_raw_integer_instead_of_action() -> None:
    with pytest.raises(TypeError, match="Action"):
        ToyGame().step(0)  # type: ignore[arg-type]


def test_reset_replays_same_action_sequence_exactly() -> None:
    game = ToyGame()
    actions = [Action.MOVE_RIGHT, Action.MOVE_LEFT, Action.WAIT]

    first = [game.step(action) for action in actions]
    game.reset()
    second = [game.step(action) for action in actions]

    assert first == second

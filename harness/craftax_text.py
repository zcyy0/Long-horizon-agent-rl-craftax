"""Native-matching text renderer for full Craftax (Craftax-Symbolic-v1).

Written from scratch against craftax core only (no craftaxlm). It mirrors
craftax.craftax.renderer.render_craftax_symbolic exactly:

  - current floor's map: state.map[player_level], a 9x11 (OBS_DIM) window,
  - a separate item layer (ladders, torches),
  - mobs (5 classes x 8 types),
  - light-map fog: tiles with light <= 0.05 are NOT observable (matching native,
    which zeroes them out). We simply omit dark tiles.

Map is rendered as one line per *visible* tile in the project's format:
    "<dy>, <dx>: [<mob> on ][<item> on ]<block>"
where (dy, dx) is the offset from the player (dy north->south, dx west->east),
followed by inventory / player / environment blocks.
"""
import numpy as np
from craftax.craftax.constants import (
    BlockType,
    ItemType,
    MAX_OBS_DIM,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
)

MOB_TYPES_PER_CLASS = 8
# obs mob-class order (see render_craftax_symbolic scan order)
MOB_CLASSES = [
    "melee_mobs",
    "passive_mobs",
    "ranged_mobs",
    "mob_projectiles",
    "player_projectiles",
]
DIRECTION_NAMES = {1: "left", 2: "right", 3: "up", 4: "down"}
POTION_COLORS = ["red", "green", "blue", "pink", "cyan", "yellow"]

# --- naming logic, reimplemented from craftax core semantics -----------------
_MOB_NAMES = {
    0: "Zombie", 1: "Gnome Warrior", 2: "Orc Soldier", 3: "Lizard", 4: "Knight",
    5: "Troll", 6: "Pigman", 7: "Frost Troll",
    8: "Cow", 9: "Bat", 10: "Snail",
    16: "Skeleton", 17: "Gnome Archer", 18: "Orc Mage", 19: "Kobold",
    20: "Archer", 21: "Deep Thing", 22: "Fire Elemental", 23: "Ice Elemental",
    24: "Arrow", 25: "Dagger", 26: "Fireball", 27: "Iceball", 28: "Arrow",
    29: "Slimeball", 30: "Fireball", 31: "Iceball",
    32: "Arrow", 33: "Dagger", 34: "Fireball", 35: "Iceball",
    36: "Arrow", 37: "Slimeball", 38: "Fireball", 39: "Iceball",
}
_MATERIAL = {1: "wood", 2: "stone", 3: "iron", 4: "diamond"}
_ENCHANT = {0: "none", 1: "fire", 2: "ice"}
_ARMOUR = {1: "iron", 2: "diamond"}
_ARMOUR_SLOTS = ["helmet", "chestplate", "leggings", "boots"]


def mob_id_to_name(mob_id):
    return _MOB_NAMES.get(int(mob_id), f"mob{int(mob_id)}")


# --- map windowing (mirrors render_craftax_symbolic) -------------------------
def _window(arr2d, player_pos, fill):
    pad = MAX_OBS_DIM + 2
    padded = np.pad(np.asarray(arr2d), pad, constant_values=fill)
    top = int(player_pos[0]) - OBS_DIM[0] // 2 + pad
    left = int(player_pos[1]) - OBS_DIM[1] // 2 + pad
    return padded[top : top + OBS_DIM[0], left : left + OBS_DIM[1]]


def light_window(state):
    lvl = int(state.player_level)
    return _window(state.light_map[lvl], np.asarray(state.player_position), 0.0) > 0.05


def block_window(state):
    lvl = int(state.player_level)
    return _window(state.map[lvl], np.asarray(state.player_position), BlockType.OUT_OF_BOUNDS.value)


def visible_block_window(state):
    """Block ids where visible (light>0.05), else -1. Matches the native obs,
    whose block one-hot is zeroed on dark tiles."""
    blocks = block_window(state)
    vis = light_window(state)
    return np.where(vis, blocks, -1)


def item_window(state):
    lvl = int(state.player_level)
    return _window(state.item_map[lvl], np.asarray(state.player_position), ItemType.NONE.value)


def visible_item_window(state):
    """Item ids where visible (light>0.05), else -1 (unseen)."""
    return np.where(light_window(state), item_window(state), -1)


def _mob_overlay(state):
    lvl = int(state.player_level)
    pp = np.asarray(state.player_position)
    half = np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
    overlay = {}
    for class_idx, cls in enumerate(MOB_CLASSES):
        m = getattr(state, cls)
        pos = np.asarray(m.position)[lvl]
        tid = np.asarray(m.type_id)[lvl]
        msk = np.asarray(m.mask)[lvl]
        for p, t, alive in zip(pos, tid, msk):
            if not bool(alive):
                continue
            loc = p - pp + half
            if (loc >= 0).all() and (loc < np.array(OBS_DIM)).all():
                mob_id = class_idx * MOB_TYPES_PER_CLASS + int(t)
                overlay[(int(loc[0]), int(loc[1]))] = mob_id_to_name(mob_id)
    return overlay


def _map_text(state):
    blocks = block_window(state)
    items = _window(state.item_map[int(state.player_level)],
                    np.asarray(state.player_position), ItemType.NONE.value)
    vis = light_window(state)
    mobs = _mob_overlay(state)
    cr, cc = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    lines = []
    for r in range(OBS_DIM[0]):
        for c in range(OBS_DIM[1]):
            dy, dx = r - cr, c - cc
            if (r, c) == (cr, cc):
                facing = DIRECTION_NAMES.get(int(state.player_direction), "?")
                lines.append(f"{dy}, {dx}: you (facing {facing})")
                continue
            if not bool(vis[r, c]):
                continue  # dark / unseen — matches native fog masking
            seg = ""
            if (r, c) in mobs:
                seg += f"{mobs[(r, c)]} on "
            it = int(items[r, c])
            if it != ItemType.NONE.value:
                seg += f"{ItemType(it).name.lower()} on "
            seg += BlockType(int(blocks[r, c])).name.lower()
            lines.append(f"{dy}, {dx}: {seg}")
    return "\n".join(lines)


def _tool(level, enchantment=None):
    mat = _MATERIAL.get(int(level), "none")
    if enchantment is not None and int(enchantment) > 0:
        return f"{mat} ({_ENCHANT.get(int(enchantment), '?')})"
    return mat


def _inventory_text(state):
    inv = state.inventory
    res = (
        f"  wood={int(inv.wood)} stone={int(inv.stone)} coal={int(inv.coal)} "
        f"iron={int(inv.iron)} diamond={int(inv.diamond)} sapphire={int(inv.sapphire)} "
        f"ruby={int(inv.ruby)} sapling={int(inv.sapling)} torches={int(inv.torches)} "
        f"arrows={int(inv.arrows)} books={int(inv.books)}"
    )
    tools = [f"pickaxe={_tool(inv.pickaxe)}",
             f"sword={_tool(inv.sword, state.sword_enchantment)}"]
    if int(inv.bow) > 0:
        tools.append(f"bow=({_ENCHANT.get(int(state.bow_enchantment), 'none')})")
    potions = np.asarray(inv.potions)
    pot = " ".join(f"{c}={int(potions[i])}" for i, c in enumerate(POTION_COLORS))
    armour = np.asarray(inv.armour)
    ench = np.asarray(state.armour_enchantments)
    worn = [
        f"{_ARMOUR_SLOTS[i]}={_ARMOUR.get(int(armour[i]), '?')}"
        + (f"({_ENCHANT[int(ench[i])]})" if int(ench[i]) > 0 else "")
        for i in range(4) if int(armour[i]) > 0
    ]
    return "\n".join([
        "# Inventory",
        res,
        "  " + " | ".join(tools),
        f"  potions: {pot}",
        f"  armour: {', '.join(worn) if worn else 'none'}",
    ])


def _player_text(state):
    return "\n".join([
        "# Player",
        f"  health={int(state.player_health)} food={int(state.player_food)} "
        f"drink={int(state.player_drink)} energy={int(state.player_energy)} "
        f"mana={int(state.player_mana)}",
        f"  xp={int(state.player_xp)} dexterity={int(state.player_dexterity)} "
        f"strength={int(state.player_strength)} intelligence={int(state.player_intelligence)}",
    ])


def _environment_text(state):
    spells = np.asarray(state.learned_spells)
    learned = [s for s, on in zip(["fireball", "iceball"], spells) if bool(on)]
    lvl = int(state.player_level)
    killed = int(np.asarray(state.monsters_killed)[lvl])
    ladder = "open" if killed >= MONSTERS_KILLED_TO_CLEAR_LEVEL else "blocked"
    return "\n".join([
        "# Environment",
        f"  floor={lvl} light_level={float(state.light_level):.2f} "
        f"sleeping={bool(state.is_sleeping)} resting={bool(state.is_resting)}",
        f"  monsters_killed_here={killed}/{MONSTERS_KILLED_TO_CLEAR_LEVEL} "
        f"(down-ladder {ladder})",
        f"  learned_spells: {', '.join(learned) if learned else 'none'}",
    ])


def render_text(state):
    """Full native-matching text observation for a Craftax (full) state."""
    return "\n".join([
        f"# Map ({OBS_DIM[0]}x{OBS_DIM[1]} egocentric; dy=N->S, dx=W->E; only visible tiles shown)",
        _map_text(state),
        _inventory_text(state),
        _player_text(state),
        _environment_text(state),
    ])

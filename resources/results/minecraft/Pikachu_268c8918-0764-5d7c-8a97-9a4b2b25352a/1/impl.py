

from pathlib import Path
import numpy as np

from helper import *
import mitsuba as mi
import traceback
import ipdb

import random
import math
import sys
import os

import mi_helper  # such that primitive call will be implemented


def main():
    from example_postprocess import parse_program
    from engine.utils.graph_utils import strongly_connected_components, get_root
    # from tu.train_setup import set_seed
    from engine.utils.train_utils import set_seed
    from dsl_utils import library, animation_func
    from minecraft_helper import execute, execute_animation

    set_seed(0)

    save_dir = Path(__file__).parent / 'renderings'
    save_dir.mkdir(exist_ok=True)

    if animation_func:
        frames = list(animation_func())
        name = animation_func.__name__
        execute_animation(frames, save_dir=(save_dir / name).as_posix(), description=name)
    else:
        exp_program_path = Path(__file__).parent / 'program.py'
        _, library_equiv = parse_program(exp_program_path.as_posix())
        scc = strongly_connected_components(library_equiv)
        print(f'{scc=}')

        try:
            root = get_root(library_equiv)
            print(f'{root=}')
        except Exception as e:
            # sometimes a function is implemented but never used, so there is no shared ancestor
            root = None
            print('[ERROR] cannot find root')
            for name, node in library_equiv.items():
                if len(node.parents) == 0 and len(node.children) > 0:
                    root = name
            if root is None:  # not sure, just pick anything?
                root = next(reversed(library.keys()))
            print(e)

        node = library_equiv[root]
        execute(node(), save_dir=(save_dir / node.name).as_posix(), description=node.name)

        save_dir = Path(__file__).parent / 'extra_renderings'
        for node in library_equiv.values():
            try:
                execute(node(), save_dir=(save_dir / node.name).as_posix(), description=node.name)
            except:
                import traceback; traceback.print_exc()
                pass


"""
Here's a program to create a simplified Pikachu shape using Minecraft blocks:

"""
from helper import *

"""
Pikachu
"""

@register()
def pikachu() -> Shape:
    return concat_shapes(
        library_call("pikachu_body"),
        library_call("pikachu_head"),
        library_call("pikachu_ears"),
        library_call("pikachu_arms"),
        library_call("pikachu_legs"),
        library_call("pikachu_tail"),
        library_call("pikachu_cheeks"),
        library_call("pikachu_eyes"),
    )

@register()
def pikachu_body() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(6, 8, 4),
            fill=True,
        ),
        translation_matrix([0, 0, 0]),
    )

@register()
def pikachu_head() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(8, 7, 6),
            fill=True,
        ),
        translation_matrix([-1, 8, -1]),
    )

@register()
def pikachu_ears() -> Shape:
    left_ear = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(2, 4, 1),
            fill=True,
        ),
        translation_matrix([-1, 15, 1]),
    )
    right_ear = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(2, 4, 1),
            fill=True,
        ),
        translation_matrix([5, 15, 1]),
    )
    return concat_shapes(left_ear, right_ear)

@register()
def pikachu_arms() -> Shape:
    left_arm = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(2, 4, 2),
            fill=True,
        ),
        translation_matrix([-2, 4, 1]),
    )
    right_arm = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(2, 4, 2),
            fill=True,
        ),
        translation_matrix([6, 4, 1]),
    )
    return concat_shapes(left_arm, right_arm)

@register()
def pikachu_legs() -> Shape:
    left_leg = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(2, 3, 2),
            fill=True,
        ),
        translation_matrix([0, -3, 1]),
    )
    right_leg = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:yellow_concrete",
            scale=(2, 3, 2),
            fill=True,
        ),
        translation_matrix([4, -3, 1]),
    )
    return concat_shapes(left_leg, right_leg)

@register()
def pikachu_tail() -> Shape:
    return concat_shapes(
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:brown_concrete",
                scale=(4, 1, 1),
                fill=True,
            ),
            translation_matrix([6, 2, 2]),
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:brown_concrete",
                scale=(2, 1, 1),
                fill=True,
            ),
            translation_matrix([8, 3, 2]),
        ),
        transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:brown_concrete",
                scale=(1, 2, 1),
                fill=True,
            ),
            translation_matrix([9, 4, 2]),
        ),
    )

@register()
def pikachu_cheeks() -> Shape:
    left_cheek = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:red_concrete",
            scale=(2, 2, 1),
            fill=True,
        ),
        translation_matrix([-1, 9, 5]),
    )
    right_cheek = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:red_concrete",
            scale=(2, 2, 1),
            fill=True,
        ),
        translation_matrix([5, 9, 5]),
    )
    return concat_shapes(left_cheek, right_cheek)

@register()
def pikachu_eyes() -> Shape:
    left_eye = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:black_concrete",
            scale=(1, 1, 1),
            fill=True,
        ),
        translation_matrix([1, 12, 5]),
    )
    right_eye = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:black_concrete",
            scale=(1, 1, 1),
            fill=True,
        ),
        translation_matrix([4, 12, 5]),
    )
    return concat_shapes(left_eye, right_eye)
"""

This program creates a simplified Pikachu shape using Minecraft blocks. Here's a breakdown of the components:

1. `pikachu()`: The main function that combines all the parts of Pikachu.
2. `pikachu_body()`: Creates the main body using yellow concrete blocks.
3. `pikachu_head()`: Creates the head using yellow concrete blocks.
4. `pikachu_ears()`: Creates two pointy ears using yellow concrete blocks.
5. `pikachu_arms()`: Creates two arms using yellow concrete blocks.
6. `pikachu_legs()`: Creates two legs using yellow concrete blocks.
7. `pikachu_tail()`: Creates a simplified lightning bolt-shaped tail using brown concrete blocks.
8. `pikachu_cheeks()`: Creates two red cheeks using red concrete blocks.
9. `pikachu_eyes()`: Creates two black eyes using black concrete blocks.

Each part is positioned relative to the body using the `translation_matrix()` function. The entire Pikachu is built from simple cuboids, creating a blocky but recognizable representation of the character.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
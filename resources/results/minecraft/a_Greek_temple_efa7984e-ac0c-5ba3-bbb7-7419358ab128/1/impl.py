

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
Here's a program to create a Greek temple using the provided `helper.py`:

"""
from helper import *

"""
a Greek temple
"""

@register()
def greek_temple() -> Shape:
    return concat_shapes(
        library_call("temple_base"),
        library_call("temple_columns"),
        library_call("temple_roof"),
        library_call("temple_steps")
    )

@register()
def temple_base() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:smooth_stone",
        scale=(20, 2, 30),
        fill=True
    )

@register()
def temple_columns() -> Shape:
    def create_column(i):
        x_offset = 2 + i * 4
        return concat_shapes(
            transform_shape(
                library_call("column"),
                translation_matrix([x_offset, 2, 2])
            ),
            transform_shape(
                library_call("column"),
                translation_matrix([x_offset, 2, 27])
            )
        )
    return loop(5, create_column)

@register()
def column() -> Shape:
    return concat_shapes(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:quartz_pillar",
            block_kwargs={"axis": "y"},
            scale=(2, 10, 2),
            fill=True
        ),
        transform_shape(
            library_call("column_capital"),
            translation_matrix([0, 10, 0])
        )
    )

@register()
def column_capital() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:chiseled_quartz_block",
        scale=(2, 1, 2),
        fill=True
    )

@register()
def temple_roof() -> Shape:
    return concat_shapes(
        library_call("roof_base"),
        library_call("roof_pediment")
    )

@register()
def roof_base() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:smooth_quartz",
            scale=(22, 2, 32),
            fill=True
        ),
        translation_matrix([-1, 12, -1])
    )

@register()
def roof_pediment() -> Shape:
    def pediment_layer(i):
        width = 22 - i * 2
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:smooth_quartz",
                scale=(width, 1, 32),
                fill=True
            ),
            translation_matrix([i, 14 + i, -1])
        )
    return loop(5, pediment_layer)

@register()
def temple_steps() -> Shape:
    def step_layer(i):
        width = 24 + i * 2
        depth = 34 + i * 2
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:stone_brick_stairs",
                block_kwargs={"facing": "north", "half": "bottom"},
                scale=(width, 1, depth),
                fill=False
            ),
            translation_matrix([-2 - i, 1 - i, -2 - i])
        )
    return loop(3, step_layer)
"""

This program creates a Greek temple with the following components:

1. A stone base
2. Ten columns (five on each side)
3. A roof with a pediment
4. Steps leading up to the temple

The temple is constructed using various Minecraft blocks to approximate the look of a Greek temple. The main structure uses quartz blocks for a white, marble-like appearance, while the base and steps use stone variants for contrast.

The temple is modular, with separate functions for each major component. The `greek_temple()` function combines all these components to create the final structure.
"""

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        extype, value, tb = sys.exc_info()
        print(e)
        print(traceback.format_exc())
        # ipdb.post_mortem(tb)
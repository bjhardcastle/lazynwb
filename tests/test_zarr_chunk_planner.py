from __future__ import annotations

import pytest

import lazynwb._zarr.chunk_planner as chunk_planner


def test_plans_bounded_1d_selection_across_chunk_boundaries() -> None:
    plan = chunk_planner._plan_array_chunks(
        array_path="/data",
        shape=(10,),
        chunks=(4,),
        selection=slice(3, 9),
    )

    assert plan.output_shape == (6,)
    assert plan.chunk_count == 3
    assert [chunk.chunk_coords for chunk in plan.chunk_reads] == [(0,), (1,), (2,)]
    assert [chunk.chunk_key for chunk in plan.chunk_reads] == [
        "data/0",
        "data/1",
        "data/2",
    ]
    assert [_selection_bounds(chunk.input_selection) for chunk in plan.chunk_reads] == [
        ((3, 4, None),),
        ((0, 4, None),),
        ((0, 1, None),),
    ]
    assert [
        _selection_bounds(chunk.output_selection) for chunk in plan.chunk_reads
    ] == [
        ((0, 1, None),),
        ((1, 5, None),),
        ((5, 6, None),),
    ]
    assert [_selection_bounds(chunk.array_selection) for chunk in plan.chunk_reads] == [
        ((3, 4, None),),
        ((4, 8, None),),
        ((8, 9, None),),
    ]


def test_plans_full_2d_read_and_edge_chunk_shapes() -> None:
    plan = chunk_planner._plan_array_chunks(
        array_path="matrix",
        shape=(5, 7),
        chunks=(2, 3),
        selection=(slice(None), slice(None)),
    )

    assert plan.output_shape == (5, 7)
    assert plan.chunk_count == 9
    assert plan.chunk_reads[0].chunk_key == "matrix/0.0"
    assert plan.chunk_reads[-1].chunk_coords == (2, 2)
    assert plan.chunk_reads[-1].chunk_key == "matrix/2.2"
    assert _selection_bounds(plan.chunk_reads[-1].input_selection) == (
        (0, 1, None),
        (0, 1, None),
    )
    assert _selection_bounds(plan.chunk_reads[-1].output_selection) == (
        (4, 5, None),
        (6, 7, None),
    )


def test_plans_separator_specific_chunk_keys() -> None:
    dot_plan = chunk_planner._plan_array_chunks(
        array_path="/root/data",
        shape=(3, 4),
        chunks=(2, 2),
        dimension_separator=".",
    )
    slash_plan = chunk_planner._plan_array_chunks(
        array_path="/root/data",
        shape=(3, 4),
        chunks=(2, 2),
        dimension_separator="/",
    )
    root_plan = chunk_planner._plan_array_chunks(
        array_path="/",
        shape=(3, 4),
        chunks=(2, 2),
        dimension_separator="/",
    )

    assert [chunk.chunk_key for chunk in dot_plan.chunk_reads] == [
        "root/data/0.0",
        "root/data/0.1",
        "root/data/1.0",
        "root/data/1.1",
    ]
    assert [chunk.chunk_key for chunk in slash_plan.chunk_reads] == [
        "root/data/0/0",
        "root/data/0/1",
        "root/data/1/0",
        "root/data/1/1",
    ]
    assert [chunk.chunk_key for chunk in root_plan.chunk_reads] == [
        "0/0",
        "0/1",
        "1/0",
        "1/1",
    ]


def test_plans_2d_output_placement_for_bounded_selection() -> None:
    plan = chunk_planner._plan_array_chunks(
        array_path="matrix",
        shape=(6, 7),
        chunks=(3, 4),
        selection=(slice(2, 5), slice(3, 7)),
    )

    assert plan.output_shape == (3, 4)
    assert [chunk.chunk_coords for chunk in plan.chunk_reads] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [_selection_bounds(chunk.input_selection) for chunk in plan.chunk_reads] == [
        ((2, 3, None), (3, 4, None)),
        ((2, 3, None), (0, 3, None)),
        ((0, 2, None), (3, 4, None)),
        ((0, 2, None), (0, 3, None)),
    ]
    assert [
        _selection_bounds(chunk.output_selection) for chunk in plan.chunk_reads
    ] == [
        ((0, 1, None), (0, 1, None)),
        ((0, 1, None), (1, 4, None)),
        ((1, 3, None), (0, 1, None)),
        ((1, 3, None), (1, 4, None)),
    ]


def test_plans_integer_index_with_reduced_output_rank() -> None:
    plan = chunk_planner._plan_array_chunks(
        array_path="matrix",
        shape=(5, 6),
        chunks=(2, 3),
        selection=(3, slice(2, 6)),
    )

    assert plan.selection == (3, slice(2, 6))
    assert plan.output_shape == (4,)
    assert [chunk.chunk_key for chunk in plan.chunk_reads] == [
        "matrix/1.0",
        "matrix/1.1",
    ]
    assert [_selection_bounds(chunk.input_selection) for chunk in plan.chunk_reads] == [
        (1, (2, 3, None)),
        (1, (0, 3, None)),
    ]
    assert [
        _selection_bounds(chunk.output_selection) for chunk in plan.chunk_reads
    ] == [
        ((0, 1, None),),
        ((1, 4, None),),
    ]


def test_marks_missing_chunks_from_available_keys_with_fill_metadata() -> None:
    plan = chunk_planner._plan_array_chunks(
        array_path="values",
        shape=(5,),
        chunks=(2,),
        available_chunk_keys={"values/0", "values/2"},
        fill_value=-1,
    )

    assert plan.has_missing_chunks
    assert plan.missing_chunk_count == 1
    assert [chunk.is_missing for chunk in plan.chunk_reads] == [False, True, False]
    assert [chunk.requires_fill for chunk in plan.chunk_reads] == [False, True, False]
    assert [chunk.fill_value for chunk in plan.chunk_reads] == [-1, -1, -1]


def test_rejects_unsupported_step_and_dimension_separator() -> None:
    with pytest.raises(ValueError, match="unit-step"):
        chunk_planner._plan_array_chunks(
            array_path="values",
            shape=(5,),
            chunks=(2,),
            selection=slice(None, None, 2),
        )

    with pytest.raises(ValueError, match="dimension_separator"):
        chunk_planner._plan_array_chunks(
            array_path="values",
            shape=(5,),
            chunks=(2,),
            dimension_separator=":",
        )


def test_empty_bounded_slice_has_no_chunk_reads() -> None:
    plan = chunk_planner._plan_array_chunks(
        array_path="values",
        shape=(8,),
        chunks=(3,),
        selection=slice(5, 2),
    )

    assert plan.output_shape == (0,)
    assert plan.chunk_reads == ()


def _selection_bounds(selection: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(
        _slice_bounds(item) if isinstance(item, slice) else item for item in selection
    )


def _slice_bounds(item: slice) -> tuple[int | None, int | None, int | None]:
    return item.start, item.stop, item.step

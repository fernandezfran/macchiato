#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Upload the collected data for each experiment."""
import os
import pathlib

import MDAnalysis as mda

import numpy as np

import pandas as pd


PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / ".data"


def for_rdf():
    """Upload data for RDF computations."""
    alloys = [
        "Li13Si64",
        "Li36Si64",
        "Li57Si64",
        "Li96Si64",
        "Li128Si64",
        "Li160Si64",
        "Li210Si64",
        "Li240Si64",
    ]

    x = np.array([13, 36, 57, 96, 128, 160, 210, 240]) / 64.0

    universes = []
    for alloy in alloys:
        with open(PATH / f"md.{alloy}.out", "r") as md_out:
            lines = md_out.readlines()

        boxes = []
        for i, line in enumerate(lines):
            if "Lattice" in line:
                lattice = [
                    list(map(float, lines[i + j + 1].split()))
                    for j in range(3)
                ]
                boxes.append(
                    np.array(
                        [
                            lattice[0][0],
                            lattice[1][1],
                            lattice[2][2],
                            90.0,
                            90.0,
                            90.0,
                        ]
                    )
                )
        boxes = np.array(boxes, dtype=object)

        u = mda.Universe(str(PATH / f"{alloy}.xyz"))
        for box, ts in zip(boxes, u.trajectory):
            u.dimensions = box

        universes.append(u)

    return x, universes


def for_gofrs():
    """Upload data for G(r) computations."""
    alloys = ["c-Si", "c-Li15Si4", "a-Si", "a-Li15Si4"]

    universes = []
    for alloy in alloys:
        with open(PATH / f"md.{alloy}.out", "r") as md_out:
            lines = md_out.readlines()

        boxes = []
        for i, line in enumerate(lines):
            if "Lattice" in line:
                lattice = [
                    list(map(float, lines[i + j + 1].split()))
                    for j in range(3)
                ]
                boxes.append(
                    np.array(
                        [
                            lattice[0][0],
                            lattice[1][1],
                            lattice[2][2],
                            90.0,
                            90.0,
                            90.0,
                        ]
                    )
                )
        boxes = np.array(boxes, dtype=object)

        u = mda.Universe(str(PATH / f"{alloy}.xyz"))
        for box, ts in zip(boxes, u.trajectory):
            u.dimensions = box

        universes.append(u)

    laaziri = pd.read_csv(PATH / "amorphous.csv")
    key = pd.read_csv(PATH / "fully_lithiated.csv")

    return alloys, universes, laaziri, key


def for_crystalline_nmr():
    alloys = ["Li12Si7", "Li7Si3", "Li13Si4", "Li15Si4"]

    x = [12 / 7, 7 / 3, 13 / 4, 15 / 4]

    boxes = np.array(
        [
            [np.array([8.531238, 19.622662, 14.312682, 90, 90, 90])],
            [np.array([7.579872, 6.564362, 18.002744, 90, 90, 120])],
            [np.array([4.420735, 7.897987, 15.011548, 90, 90, 90])],
            [np.array([10.566048, 10.566048, 10.566048, 90, 90, 90])],
        ],
        dtype=object,
    )
    universes = []
    for box, alloy in zip(boxes, alloys):
        u = mda.Universe(str(PATH / f"{alloy}.xyz"))
        for ts in u.trajectory:
            u.dimensions = box

        universes.append(u)

    experimental_data = [str(PATH / f"{alloy}.csv") for alloy in alloys]

    return x, universes, experimental_data

# Importing some libraries
import networkx as nx
import pandas as pd
import numpy as np
import math as mt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygraphviz as pgv
import queue as qu
import json
from IPython.display import Image

########## Importing COVID-19 Data
with open('Resources/dpc-covid19-ita-province.json') as f:
    d = json.load(f)

# Create a DataFrame with COVID data, we need just some columns
city_dataframe = pd.DataFrame(d)[['sigla_provincia', 'lat', 'long']].drop_duplicates()
city_dataframe.fillna({'sigla_provincia': "None", 'lat': 0, 'long': 0}, inplace=True)

print("Dataframe contains " + str(city_dataframe.count()[0]) + " rows")

# Remove data having latitude = 0 or longitude = 0 or provincia = "In fase di definizione/aggiornamento"
city_dataframe.drop(city_dataframe[(city_dataframe['lat'] == 0) | \
                                   (city_dataframe['long'] == 0) | \
                                   (city_dataframe['sigla_provincia'] == 'None') \
                                   ].index, inplace=True)

city_dataframe.reset_index(drop=True, inplace=True)

print("After removing unusable data, Dataframe contains " + str(city_dataframe.count()[0]) + " rows")


# If we don't want to use Dataframes we can use 2 dictionaries with key = sigla_provincia
# and value = long / lat. In this way we can loop over cities and dictionary will check if that city
# has already been inserted in O(1). Afterthat it can be useful to convert those dictionaries to
# list to sort them.
# In following implementation we will use p_dataframe approach.

########## Algorithms
# We are not explicitly adding nodes to p_graph --> nodes without any edge will not be put in the p_graph
# In all those implementation we consider a p_dataframe having 3 columns:
#   0) ID
#   1) X Position
#   2) Y Position

# Note: Functions do not return graph but modify the input graph because assignment
# has some problems with %%timeit in notebook.

##### a) Naive algorithm: iteration over all couples --> Cost: O(n^2)
def all_couple_edges(p_graph, p_dataframe, p_radius):
    # O (n)
    for i in p_dataframe.index:
        # O (n)
        for j in p_dataframe.index:
            # We do not want edges from a city to itself
            if i != j and \
                    p_dataframe.iloc[i, 1] - p_radius <= p_dataframe.iloc[j, 1] <= p_dataframe.iloc[i, 1] + p_radius and \
                    p_dataframe.iloc[i, 2] - p_radius <= p_dataframe.iloc[j, 2] <= p_dataframe.iloc[i, 2] + p_radius:
                p_graph.add_edge(p_dataframe.iloc[i, 0], p_dataframe.iloc[j, 0])


##### b) Binary search on ordered p_dataframe --> Cost: O(n * log(n))
## Utility function
# Given a p_dataframe with 2 columns:
#   0) ID
#   1) Position
# returns a set of all ID couples within p_radius distance
# We will not search all matching cities (basing on p_radius) but just the "least"
# city (the one with least Position) within p_radius distance.
def binary_search_single(p_dataframe, p_radius):
    # Edges between near cities basing on x position
    # Use of dictionary, in this way search of an element costs O(1)
    edges = {}

    # Sort p_dataframe basing on position O (n log n) using quicksort
    # Use of tmp_dataframe to leave p_dataframe as received
    tmp_dataframe = p_dataframe.sort_values(by=p_dataframe.columns[1])
    tmp_dataframe.reset_index(drop=True, inplace=True)

    # O(n)
    for i in tmp_dataframe.index:
        # Set pointers to be used in iterative binary search
        first = 0
        # We just check the left half because we do not need double
        # couples (a, b) and (b, a) since we will use undirect p_graph
        last = i - 1
        found = False

        # O (log n)
        while first <= last and not found:
            midpoint = (first + last) // 2

            # Check if element at midpoint position is near enough
            if tmp_dataframe.iloc[i, 1] - p_radius <= tmp_dataframe.iloc[midpoint, 1]:

                # If element at midpoint position is the leftmost element within p_radius distance
                # i.e. element at (midpoint - 1) position is too far
                if midpoint == 0 or tmp_dataframe.iloc[i, 1] - p_radius > tmp_dataframe.iloc[midpoint - 1, 1]:

                    # We add to edges all couples composed by (element at i position, element at j position)
                    # for all j from midpoint to i (excluded)
                    edges.update([((tmp_dataframe.iloc[i, 0], tmp_dataframe.iloc[j, 0]), None)
                                  for j in range(midpoint, i)])
                    found = True

                # Otherwise (element at (midpoint - 1) position is near enough)
                # We search in left half
                else:
                    last = midpoint - 1

            # Otherwise we must search in right half
            else:
                first = midpoint + 1

    return edges


def binary_search_edges(p_graph, p_dataframe, p_radius):
    x_edges = binary_search_single(p_dataframe, p_radius)
    y_edges = binary_search_single(p_dataframe.iloc[:, 0::2], p_radius)

    # O(n)
    for k in x_edges.keys():
        # Searching both for (a,b) and (b,a)
        # O(1)
        if k in y_edges or k[::-1] in y_edges:
            p_graph.add_edge(*k)


##### c) Binary search on ordered p_dataframe --> Cost: O(n * log(n))
# We can improve previous algo if we consider that we can try to guess where
# the "least" city is (assuming that cities are almost uniformly distributed).
# In this way we don't have to search in all previous elements but we can
# reduce them in advance.
def smart_binary_search_single(p_dataframe, p_radius):
    # Edges between near cities basing on x position
    # Use of dictionary, in this way search of an element costs O(1)
    edges = {}

    # Sort p_dataframe basing on position O (n log n) using quicksort
    # Use of tmp_dataframe to leave p_dataframe as received
    tmp_dataframe = p_dataframe.sort_values(by=p_dataframe.columns[1])
    tmp_dataframe.reset_index(drop=True, inplace=True)

    # We use a factor of 3 to wide the area in which we guess that the "least" city is
    j = 3 * p_radius / (tmp_dataframe.iloc[-1, 1] - tmp_dataframe.iloc[0, 1]) * (tmp_dataframe.count()[0] - 1)
    jump = int(j)

    # O(n)
    for i in tmp_dataframe.index:
        # Set pointers to be used in iterative binary search
        first = i - jump

        # We set first to 0 as well (search will be done in all i - 1 previous elements) if:
        #   - first is < 0, i.e. i is in first elements
        #   - our "jump" backward ended again within p_radius distance.
        #     In that case we can decide to try another jump backward and loop in that way until
        #     [first] is outsite the p_radius. To avoid loops of jump and check in those
        #     rare cases it's easier (and maybe faster) searching in all previous elements.
        if first < 0 or tmp_dataframe.iloc[i, 1] - p_radius < tmp_dataframe.iloc[first, 1]:
            first = 0

        # We just check the left half because we do not need double couples (a, b) and (b, a).
        last = i - 1
        found = False

        # O (log n)
        while first <= last and not found:
            midpoint = (first + last) // 2

            # Check if element at midpoint position is near enough
            if tmp_dataframe.iloc[i, 1] - p_radius <= tmp_dataframe.iloc[midpoint, 1]:

                # If element at midpoint position is the leftmost element within p_radius distance
                # i.e. element at (midpoint - 1) position is too far
                if midpoint == 0 or \
                        tmp_dataframe.iloc[i, 1] - p_radius > tmp_dataframe.iloc[midpoint - 1, 1]:

                    # We add to edges all couples composed by (element at i position, element at j position)
                    # for all j from midpoint to i (excluded)
                    edges.update([((tmp_dataframe.iloc[i, 0], tmp_dataframe.iloc[j, 0]), None)
                                  for j in range(midpoint, i)])
                    found = True

                # Otherwise (element at (midpoint - 1) position is near enough)
                # We search in left half
                else:
                    last = midpoint - 1

            # Otherwise we must search in right half
            else:
                first = midpoint + 1

    return edges


def smart_binary_search_edges(p_graph, p_dataframe, p_radius):
    x_edges = smart_binary_search_single(p_dataframe, p_radius)
    y_edges = smart_binary_search_single(p_dataframe.iloc[:, 0::2], p_radius)

    # O(n)
    for k in x_edges.keys():
        # Searching both for (a,b) and (b,a)
        # O(1)
        if k in y_edges or k[::-1] in y_edges:
            p_graph.add_edge(*k)


##### d) "Range" scan
# We consider a range of elements within p_radius.
def scan(p_dataframe, p_radius):
    edges = {}

    tmp_dataframe = p_dataframe.sort_values(by=p_dataframe.columns[1])
    tmp_dataframe.reset_index(drop=True, inplace=True)

    # Range starting index
    i = 0
    # Range ending index
    j = 0
    last_j = 0
    dataframe_count = tmp_dataframe.count()[0]

    while j < dataframe_count:

        # If last city in current range (j) isn't far enough
        if tmp_dataframe.iloc[i, 1] + p_radius >= tmp_dataframe.iloc[j, 1]:
            # While we haven't reached the end of the list or
            # current element isn't the greatest within p_radius
            while j < dataframe_count - 1 and \
                    tmp_dataframe.iloc[i, 1] + p_radius >= tmp_dataframe.iloc[j + 1, 1]:
                # Add an element to current range
                j += 1

            # Add to edges couples of all elements in current range. We add all combinations with
            # left element < right element
            edges.update([((tmp_dataframe.iloc[left_elem, 0], tmp_dataframe.iloc[right_elem, 0]), None)
                          for left_elem in range(i, j)
                          for right_elem in range(max(last_j, left_elem) + 1, j + 1)])

            # We save a pointer to remember current threshold of the range
            last_j = j

            # If we reached the end of the list
            if j == dataframe_count - 1:
                break

        # Otherwise move the range of 1 element
        j += 1
        i += 1

        # We "remove" from current range (incrementing i) all elements that are too far
        # from greatest city of the range (j)
        while i < j and tmp_dataframe.iloc[i, 1] + p_radius < tmp_dataframe.iloc[j, 1]:
            i += 1
    return edges


def scan_edges(p_graph, p_dataframe, p_radius):
    x_edges = scan(p_dataframe, p_radius)
    y_edges = scan(p_dataframe.iloc[:, 0::2], p_radius)

    # O(n)
    for k in x_edges.keys():
        # Searching both for (a,b) and (b,a)
        # O(1)
        if k in y_edges or k[::-1] in y_edges:
            p_graph.add_edge(*k)


# Create a graph variable for each algorithm
P_a = nx.Graph()
P_b = nx.Graph()
P_c = nx.Graph()
P_d = nx.Graph()
radius = 0.8

################################################################ %%timeit
P_a.clear()
all_couple_edges(P_a, city_dataframe, radius)

################################################################ %%timeit
P_b.clear()
binary_search_edges(P_b, city_dataframe, radius)

################################################################ %%timeit
P_c.clear()
smart_binary_search_edges(P_c, city_dataframe, radius)

################################################################ %%timeit
P_d.clear()
scan_edges(P_d, city_dataframe, radius)

# Checking that all algo give the same result
if P_a.nodes != P_b.nodes:
    raise Exception("P_b.nodes != P_a.nodes")

if P_a.edges != P_b.edges:
    raise Exception("P_b.edges != P_a.edges")

if P_a.nodes != P_c.nodes:
    raise Exception("P_c.nodes != P_a.nodes")

if P_a.edges != P_c.edges:
    raise Exception("P_c.edges != P_a.edges")

if P_a.nodes != P_d.nodes:
    raise Exception("P_d.nodes != P_a.nodes")

if P_a.edges != P_d.edges:
    raise Exception("P_d.edges != P_a.edges")
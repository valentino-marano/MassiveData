# Importing some libraries
import networkx as nx
import pandas as pd
import timeit as tm
import numpy as np
import math as mt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygraphviz as pgv
import queue as qu
import json
from PIL import Image
from random import choice


##### Set viz to False to use matplotlib to plot graphs,
# otherwise set it to True to use pygraphviz
viz = False

########## Importing COVID-19 Data
with open('Resources/dpc-covid19-ita-province.json') as f:
    d = json.load(f)

# Create a DataFrame with COVID data, we need just some columns
city_dataframe = pd.DataFrame(d)[['sigla_provincia', 'lat', 'long']].drop_duplicates()
city_dataframe.fillna({'sigla_provincia': "None", 'lat': 0, 'long': 0}, inplace=True)

print("Dataframe contains " + str(city_dataframe.count()[0]) + " rows")

# Remove data having latitude = 0 or longitude = 0 or provincia = "In fase di definizione/aggiornamento"
city_dataframe.drop(city_dataframe[(city_dataframe['lat'] == 0) |
                                   (city_dataframe['long'] == 0) |
                                   (city_dataframe['sigla_provincia'] == 'None')
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

# Note: Functions do not return p_graph but modify the input p_graph because assignment
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


# Create a p_graph variable for each algorithm
P_all_couples = nx.Graph()
P = nx.Graph()
radius = 0.8


# Defining a function to measure execution time
def perf_time(p_exe):
    min_time = None
    max_time = None
    avg_time = 0
    attempts = 1

    for i in range(attempts):
        start = tm.default_timer()
        exec(p_exe)
        time = tm.default_timer() - start
        if min_time is None or min_time > time:
            min_time = time
        if max_time is None or max_time < time:
            max_time = time
        avg_time += time
    avg_time /= attempts
    print("\nExecuted: " + p_exe + "\nMin time: " + str(min_time) + ", Max time: " + str(max_time) +
          ", avg_time: " + str(avg_time))


# Defining a function to execute time measures using all our functions
def exec_perf(p_graph_all, p_graph, p_dataframe, p_radius):
    if p_dataframe == "points_dataframe":
        print("\nExecution of algo all_couple_edges has been skipped because it takes too much time.\n" +
              "Executed just once with result: time: 74.25751607200073")
    else:
        perf_time(
            p_graph_all + ".clear() ; all_couple_edges(" + p_graph_all + ", " + p_dataframe + ", " + p_radius + ")")

    for func in ("binary_search_edges", "smart_binary_search_edges", "scan_edges"):
        perf_time(p_graph + ".clear() ; " + func + "(" + p_graph + ", " + p_dataframe + ", " + p_radius + ")")


def check_result_graphs(p_graph_all, p_graph):
    # Checking that all algo give the same result
    if eval(p_graph_all + ".nodes != " + p_graph + ".nodes"):
        raise Exception(p_graph_all + ".nodes != " + p_graph + ".nodes")
    if eval(p_graph_all + ".edges != " + p_graph + ".edges"):
        raise Exception(p_graph_all + ".edges != " + p_graph + ".edges")


exec_perf("P_all_couples", "P", "city_dataframe", "radius")
check_result_graphs("P_all_couples", "P")

########## Generate 2000 pairs of double(x, y)
radius = 0.08
R = nx.Graph()
xMin = 30
xMax = 50
yMin = 10
yMax = 20
couples_count = 2000
points_dataframe = pd.DataFrame()

# Generate column x with couples_count rows of elements in [xMin, xMax)
points_dataframe['x'] = np.random.random_sample(couples_count) * (xMax - xMin) + xMin

# Generate column y with couples_count rows of elements in [yMin, yMax)
points_dataframe['y'] = np.random.random_sample(couples_count) * (yMax - yMin) + yMin

# Generate column label with couples_count rows of (x value, y value)
points_dataframe['label'] = "(" + points_dataframe['x'].astype(str) + ", " + \
                            points_dataframe['x'].astype(str) + ")"
# Reorder columns
points_dataframe = points_dataframe[['label', 'x', 'y']]

# Replace duplicates to have clean data
# We just change y value and check that new couple is unique
for dup_ind in points_dataframe[points_dataframe.duplicated()].index:
    while (points_dataframe['label'].value_counts()[points_dataframe.loc[dup_ind, 'label']]) > 1:
        points_dataframe.loc[dup_ind, 'y'] = np.random.random_sample() * (yMax - yMin) + yMin
        points_dataframe.loc[dup_ind, 'label'] = "(" + str(points_dataframe.loc[dup_ind, 'x']) + \
                                                 ", " + str(points_dataframe.loc[dup_ind, 'y']) + ")"

exec_perf("P_all_couples", "R", "points_dataframe", "radius")


########## Weight graphs
##### Utility function
def weight_graph(p_graph, p_dataframe):
    for edge in p_graph.edges:
        p_graph.edges[edge]['weight'] = mt.sqrt(((p_dataframe.loc[p_dataframe.iloc[:, 0] == edge[0]].iloc[0, 1]) -
                                                 (p_dataframe.loc[p_dataframe.iloc[:, 0] == edge[1]].iloc[0, 1])) ** 2 +
                                                ((p_dataframe.loc[p_dataframe.iloc[:, 0] == edge[0]].iloc[0, 2]) -
                                                 (p_dataframe.loc[p_dataframe.iloc[:, 0] == edge[1]].iloc[0, 2])) ** 2)


weight_graph(P, city_dataframe)
weight_graph(R, points_dataframe)


########## Eulerian Path
# From Wikipedia (https://en.wikipedia.org/wiki/Eulerian_path):
# In p_graph theory, an Eulerian trail (or Eulerian path) is a trail in a finite p_graph that visits every edge
# exactly once (allowing for revisiting vertices). Similarly, an Eulerian circuit or Eulerian cycle is an
# Eulerian trail that starts and ends on the same vertex.

##### Defining a generic function to draw graphs having an "order" p_attribute on edges.
# Edges will have different color basing on visit order and a label with visit order.

def draw_graph_viz(p_graph, p_attribute=None, p_title=None):
    cmap = plt.cm.get_cmap('Blues')
    norm = mpl.colors.Normalize(vmin=-2, vmax=len(p_graph.edges))

    # We want p_graph to be directed only if p_attribute != None
    # Otherwise it means we are drawing the original p_graph (unidrect)
    gr = pgv.AGraph(directed=(p_attribute is not None))

    for start, end, attr in p_graph.edges(data=True):
        if p_attribute is not None:
            gr.add_edge(start, end, color=str(mpl.colors.to_hex(cmap(norm(attr[p_attribute])))),
                        label=" " + str(attr[p_attribute]))
        else:
            gr.add_edge(start, end)

    for node in p_graph.nodes():
        gr.add_node(node)

    gr.layout(prog='dot')
    gr.draw("file.png")
    image = Image.open("file.png")
    # We also print the title because some viewers cannot show title
    print(p_title)
    image.show(title=p_title)


# input("Press any key to continue...")


def draw_graph(p_graph, p_attribute=None, p_figsize=None):
    plt.figure(figsize=p_figsize)
    cmap = plt.cm.Blues
    font_size = 25
    pos = nx.circular_layout(p_graph)
    node_color = '#A0CBE2'
    node_size = 1500

    if p_attribute is not None:
        # Retrieving labels to be put on edges
        edge_labels = dict([((start, end,), attr[p_attribute])
                            for start, end, attr in p_graph.edges(data=True)])
        # Setting different colors on edges basing on p_attribute value
        colors = [attr[p_attribute] for start, end, attr in p_graph.edges(data=True)]

        nx.draw_networkx_edge_labels(p_graph, pos=pos, edge_labels=edge_labels, font_size=font_size)

        if len(colors) > 0:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
            plt.colorbar(sm, ticks=colors)

    else:
        colors = node_color

    nx.draw(p_graph, with_labels=True, pos=pos,
            node_color=node_color, font_size=font_size, node_size=node_size, font_color='w', \
            edge_color=colors, width=4, edge_cmap=cmap)
    plt.show()


#### Exception definition
# These exceptions will be used to manage eventual limitation of following algorithms
class NotConnectedGraph(Exception):
    def __init__(self, message):
        self.message = message


class TooManyOddNodes(Exception):
    def __init__(self, message):
        self.message = message


#### Fleury's algorithm
# Fleury's algorithm is an elegant but inefficient algorithm.
# 1. Check that p_graph has all edges in the same component
# 2. Check that p_graph has at most 2 vertices of odd degree
# 3. Choose a vertex of odd degree, if the p_graph has none choose an arbitrary vertex.
#   3.1 Choose next edge in the path to be one whose deletion would not disconnect the p_graph.
#       If there is no such edge pick the remaining edge left at the current vertex.
#   3.2 Use this edge to reach the other node and delete the edge.
#   3.3 If current vertex has no more edges it means that the Graph has no more edge.
#       Otherwise return to 3.1
#
# p_viz param allow to decide if use pygraphviz and show images of graphs
# or use matplotlib to show graphs.
def fleury(p_graph, p_viz=True):
    # p_graph that will represent with direct edges
    # the eulerian trail
    trail = nx.DiGraph()

    # Check if p_graph is connected
    if not nx.is_connected(p_graph):
        raise NotConnectedGraph("Graph is not connected")

    # Use of tmp_graph to leave p_graph as received
    tmp_graph = p_graph.copy()

    # Check which nodes have odd degree and raise exception if more than 2 have been found
    odd_degree_nodes = []
    for node in tmp_graph.nodes:
        if tmp_graph.degree(node) % 2 != 0:
            odd_degree_nodes.append(node)
            if len(odd_degree_nodes) > 2:
                raise TooManyOddNodes("Graph has at least 3 nodes with odd degree: " + str(odd_degree_nodes))

    if len(odd_degree_nodes) == 1:
        raise TooManyOddNodes("Graph has 1 node with odd degree, it should have 0 or 2")

    if p_viz:
        draw_graph_viz(p_graph, p_title="Original p_graph")
    else:
        draw_graph(p_graph, p_figsize=(6, 6))

    # Start with a node with odd degree (if any)
    if len(odd_degree_nodes) > 0:
        node = odd_degree_nodes[0]
    else:
        node = choice(list(tmp_graph.nodes))

    end = False
    trail.add_node(node)
    order = 1
    while not end:
        # Search non-bridge edges
        not_bridge_edges = {*tmp_graph.edges(node)} - {*nx.bridges(tmp_graph, node)}
        if len(not_bridge_edges) > 0:
            next_node = not_bridge_edges.pop()[1]

        # If node has at least 1 neighbour
        elif len(tmp_graph[node]) > 0:
            next_node = [*tmp_graph[node]][0]

        else:
            end = True

        if not end:
            trail.add_edge(node, next_node, order=order)
            tmp_graph.remove_edge(node, next_node)
            node = next_node
            if p_viz:
                draw_graph_viz(trail, "order", "Step #" + str(order))
            else:
                draw_graph(trail, "order")
            order = order + 1


# Utility function to manage particulare cases of graphs of cities and points.
def run(p_graph, p_fun, p_viz=True):
    try:
        p_fun(p_graph, p_viz=p_viz)
    except TooManyOddNodes as exc:
        print(str(exc))
    except NotConnectedGraph:
        print("Graph is not connected, algorithm will be applied on single connected components")
        for nodes in list(nx.connected_components(p_graph)):
            print("-------------------------------")
            print("Connected component: " + str(nodes))
            conn_comp_graph = nx.subgraph(p_graph, nodes)
            if len(nodes) <= 3:
                print("Connected component has just " + str(len(nodes)) + " nodes, Eulerian trail is trivial")
            else:
                try:
                    p_fun(conn_comp_graph, p_viz=p_viz)
                except TooManyOddNodes as exc:
                    print("This connected component will be skipped")
                    print(str(exc))


# An example with a dummy p_graph
Ex = nx.Graph()
Ex.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
fleury(Ex, p_viz=viz)

# Call the algo on city graph P and point graph R
run(P, fleury, p_viz=viz)
run(R, fleury, p_viz=viz)


##### Hierholzer's algorithm
# Hierholzer's is more efficient than Fleury's algorithm.
# 1. Choose any starting vertex v
# 2. Follow a trail of edges from that vertex until returning to v.
#    It is not possible to get stuck at any vertex other than v, because the even degree of all vertices
#    ensures that, when the trail enters another vertex w there must be an unused edge leaving w. The tour
#    formed in this way is a closed tour, but may not cover all the vertices and edges of the initial p_graph.
# 3. As long as there exists a vertex u that belongs to the current tour but that has adjacent edges not
#    part of the tour, start another trail from u, following unused edges until returning to u, and join the
#    tour formed in this way to the previous tour.
#
# Since we assume the original p_graph is connected, repeating the previous step will exhaust all edges of the
# p_graph.

# Utility Class Double Linked List
# In this way add and pop at begin or end will cost O(1)
class Node:
    def __init__(self, initdata):
        self.data = initdata
        self.next = None
        self.prev = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def getPrev(self):
        return self.prev

    def setData(self, newdata):
        self.data = newdata

    def setNext(self, next):
        self.next = next

    def setPrev(self, prev):
        self.prev = prev

    def __str__(self):
        return str(self.data)


class DoubleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def __iter__(self):
        return DoubleLinkedListIterator(self)

    def isEmpty(self):
        return self.head is None

    def add_begin(self, item):
        temp = Node(item)
        if self.isEmpty():
            self.tail = temp
        else:
            self.head.setPrev(temp)
            temp.setNext(self.head)
        self.head = temp

    def add_end(self, item):
        temp = Node(item)
        if self.isEmpty():
            self.head = temp
        else:
            self.tail.setNext(temp)
            temp.setPrev(self.tail)
        self.tail = temp

    def pop_begin(self):
        temp = self.head
        self.head = temp.next
        # If list is empty after pop
        if self.head is None:
            self.tail = None
        else:
            self.head.setPrev(None)
        return temp.getData()

    def pop_end(self):
        temp = self.tail
        self.tail = temp.prev
        # If list is empty after pop
        if self.tail is None:
            self.head = None
        else:
            self.tail.setNext(None)
        return temp.getData()

    def __str__(self):
        ret_str = "head --> "
        for node in self:
            if ret_str != "head --> ":
                ret_str += " <--> "
            ret_str += str(node)
        ret_str += " <-- tail"
        return ret_str


class DoubleLinkedListIterator:
    def __init__(self, double_linked_list):
        self.doubleLinkedList = double_linked_list
        self.pointer = double_linked_list.head

    def __next__(self):
        if self.pointer is None:
            raise StopIteration
        else:
            node = self.pointer
            self.pointer = self.pointer.next
            return node


def hierholzer(p_graph, p_viz=True):
    # Directed Graph that will contain Eulerian path
    trail = nx.DiGraph()

    # Double Linked List that will contain current path
    curr_path = DoubleLinkedList()

    # Check if p_graph is connected
    if not nx.is_connected(p_graph):
        raise NotConnectedGraph("Graph is not connected")

    # Use of tmp_graph to leave p_graph as received
    tmp_graph = p_graph.copy()

    # Check which nodes have odd degree and raise exception if more than 2 have been found
    odd_degree_nodes = []
    for node in tmp_graph.nodes:
        if tmp_graph.degree(node) % 2 != 0:
            odd_degree_nodes.append(node)
            if len(odd_degree_nodes) > 2:
                raise TooManyOddNodes("Graph has at least 3 nodes with odd degree: " + str(odd_degree_nodes))

    if len(odd_degree_nodes) == 1:
        raise TooManyOddNodes("Graph has 1 node with odd degree, it should have 0 or 2")

    if p_viz:
        draw_graph_viz(p_graph, p_title="Original p_graph")
    else:
        draw_graph(p_graph, p_figsize=(6, 6))

    # Start with an odd node (if any)
    if len(odd_degree_nodes) > 0:
        node = odd_degree_nodes[0]
    else:
        node = choice(list(tmp_graph.nodes))

    end = False
    order = 1
    curr_path.add_end(node)
    prev_node = None
    while not end:
        # If node has at least 1 neighbour
        if len(tmp_graph[node]) > 0:
            next_node = [*tmp_graph[node]][0]

        else:
            # If we visited all edges we just need to add to DiGraph
            # all edges from curr_path
            if len(tmp_graph.edges()) == 0:
                end = True
                while not curr_path.isEmpty():
                    node = curr_path.pop_end()
                    trail.add_node(node)
                    if prev_node is not None:
                        trail.add_edge(prev_node, node, order=len(trail.edges()) + 1)
                        if p_viz:
                            draw_graph_viz(trail, "order", "Step #" + str(order))
                        else:
                            draw_graph(trail, "order")
                    prev_node = node
            # If there are still some edges in the Graph
            else:
                found = False
                end = True
                # We search within all already visited nodes in curr_path
                while not curr_path.isEmpty() and not found:
                    node = curr_path.pop_end()
                    # If extracted node from curr_path has at least a non-visited edge
                    # we set that node as next_node
                    if len(tmp_graph[node]) > 0:
                        next_node = [*tmp_graph[node]][0]
                        curr_path.add_end(node)
                        found = True
                        end = False
                    # Otherwise we add that node to trail
                    else:
                        if prev_node is None:
                            prev_node = node
                            trail.add_node(prev_node)
                        else:
                            trail.add_edge(prev_node, node, order=len(trail.edges()) + 1)
                            if p_viz:
                                draw_graph_viz(trail, "order", "Step #" + str(order))
                            else:
                                draw_graph(trail, "order")
                            prev_node = node

        if not end:
            tmp_graph.remove_edge(node, next_node)
            curr_path.add_end(next_node)
            node = next_node

        order = order + 1


Ex = nx.Graph()
Ex.add_nodes_from(range(1, 7))
Ex.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 6), (2, 5), (2, 6), (2, 3), (3, 6), (4, 3), (4, 5), (5, 6)])
hierholzer(Ex, p_viz=viz)

Ex = nx.Graph()
Ex.add_nodes_from("abcdef")
Ex.add_edges_from([('a', 'b'), ('a', 'c'), ('d', 'b'), ('d', 'c'), ('e', 'c'), ('e', 'f'), ('c', 'f')])
hierholzer(Ex, p_viz=viz)

# Call the algo on city graph P and point graph R
run(P, hierholzer, p_viz=viz)
run(R, hierholzer, p_viz=viz)


########## Eccentricity Centrality
# From Wikipedia page: https: // en.wikipedia.org / wiki / Distance_(graph_theory)
# A peripheral vertex in a graph of diameter d is one that is distance d from some other vertex.
# Formallly: v is peripheral <--> ecc(v) = d
# A pseudo - peripheral vertex has the property that for any vertex u, if v is as far away from u as possible,
# then u is as far away from v as possible.
# Formally: u is pseudo - peripheral <--> forall v such that d(u, v) = ecc(u) holds ecc(u) = ecc(v).

##### Algorithm
# 1. Choose a vertex u.
# 2. Among all the vertices that are as far from u as possible, let v be one with minimal degree.
# 	2.1. If ecc(v) > ecc(u) --> set u = v and return to step 2
# 	2.2.Else u is a pseudo - peripheral vertex.

# Utility function to execute a BFS and return:
#   - eccentricity of p_start_node
#   - node at maximum distance having minimal degree
def bfs_peripheral(p_graph, p_start_node):
    bfs_que = qu.SimpleQueue()
    bfs_que.put((p_start_node, 0))
    max_dist_node = None
    eccentricity = -1
    queued = {p_start_node: None}
    # O (n)
    while not bfs_que.empty():
        (act_node, dist) = bfs_que.get()
        leaf = True
        for node in p_graph[act_node]:
            if node not in queued:
                if leaf:
                    leaf = False
                queued[node] = None
                bfs_que.put((node, dist + 1))

        if leaf and (dist > eccentricity or (dist == eccentricity and
                                             p_graph.degree[max_dist_node] > p_graph.degree[act_node])):
            max_dist_node = act_node
            eccentricity = dist

    return eccentricity, max_dist_node


def pseudo_peripheral(p_graph, p_start_node=None):
    if p_start_node is None:
        p_start_node = [*p_graph.nodes()][0]

    u = p_start_node
    eccentricity_u, v = bfs_peripheral(p_graph, u)
    print("u = " + str(u) + ", ecc(" + str(u) + ") = " + str(eccentricity_u) + ", v = " + str(v))
    while True:
        eccentricity_v, z = bfs_peripheral(p_graph, v)
        print("ecc(" + str(v) + ") = " + str(eccentricity_v) + ", z = " + str(z))
        if eccentricity_v == eccentricity_u:
            return u

        eccentricity_u = eccentricity_v
        u = v
        v = z


Ex = nx.Graph()
Ex.add_nodes_from(range(1, 5))
Ex.add_edges_from([(1, 2), (1, 4), (2, 3), (2, 3), (2, 4), (3, 4)])
draw_graph(Ex)
pseudo_peripheral(Ex)

Ex = nx.Graph()
Ex.add_nodes_from(range(1, 9))
Ex.add_edges_from([(6, 4), (4, 2), (4, 8), (2, 1), (1, 3), (3, 5), (5, 7), (5, 8)])
draw_graph(Ex)
print(pseudo_peripheral(Ex))
print("--------------")
print(pseudo_peripheral(Ex, 8))

import sys
import puzz
import pdqpq


GOAL_STATE = puzz.EightPuzzleBoard("012345678")


def solve_puzzle(start_state, flavor):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state (EightPuzzleBoard): the start state for the search
        flavor (str): tag that indicate which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
        'path' - list of 2-tuples representing the path from the start to the goal state (both 
            included).  Each entry is a (str, EightPuzzleBoard) pair indicating the move and 
            resulting successor state for each action.  Omitted if the search fails.
        'path_cost' - the total cost of the path, taking into account the costs associated with 
            each state transition.  Omitted if the search fails.
        'frontier_count' - the number of unique states added to the search frontier at any point 
            during the search.
        'expanded_count' - the number of unique states removed from the frontier and expanded 
            (successors generated)

    """
    if flavor.find('-') > -1:
        strat, heur = flavor.split('-')
    else:
        strat, heur = flavor, None

    if strat == 'bfs':
        return BreadthFirstSolver().solve(start_state)
    elif strat == 'ucost':
        return UniformCostSolver().solve(start_state)  # delete this line!
    elif strat == 'greedy':
        return GreedySolver().solve(start_state, heur) # delete this line!
    elif strat == 'astar':
        return AStarSolver().solve(start_state, heur)  # delete this line!
    else:
        raise ValueError("Unknown search flavor '{}'".format(flavor))


class BreadthFirstSolver:
    """Implementation of Breadth-First Search based puzzle solver"""

    def __init__(self):
        self.goal = GOAL_STATE
        self.parents = {}  # state -> parent_state
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        self.frontier_count = 0  # increment when we add something to frontier
        self.expanded_count = 0  # increment when we pull something off frontier and expand
    
    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """

        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node

                    # BFS checks for goal state _before_ adding to frontier
                    if succ == self.goal:
                        return self.get_results_dict(succ)
                    else:
                        self.add_to_frontier(succ)

        # if we get here, the search failed
        return self.get_results_dict(None) 

    def add_to_frontier(self, node):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def get_results_dict(self, state):
        """Construct the output dictionary for solve_puzzle()
        
        Args:
            state (EightPuzzleBoard): final state in the search tree
        
        Returns:
            A dictionary describing the search performed (see solve_puzzle())

        """
        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def get_path(self, state):
        """Return the solution path from the start state of the search to a target.
        
        Results are obtained by retracing the path backwards through the parent tree to the start
        state for the serach at the root.
        
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            A list of EightPuzzleBoard objects representing the path from the start state to the
            target state

        """
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 

        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path

        """
        cost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y)        
            cost += int(tile)**2
        return cost

class UniformCostSolver(BreadthFirstSolver):
    def __init__(self):
        super().__init__()
        #switch to a priority queue for UCS
        self.frontier = pdqpq.PriorityQueue()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():

            node = self.frontier.pop()  # get the next node in the frontier queue
            #successor states to current node, also adds node to list of explored states
            succs = self.expand_node(node)
            if node == self.goal:
                return self.get_results_dict(node)

            #move is a stirng which is either left, right, up, or down, successor is the (max 4) successor boards for those 4 moves
            for move, succ in succs.items():
                alt_cost = self.get_cost(node) + self._transition_cost(node, move)
                #if the current successor has already been explored
                if (succ in self.explored):
                    #if the successor has already been explored, not possible to get a cheaper path from a higher priority node
                    continue

                #current successor has not been explored
                else:
                    #check to see if the frontier cost is cheapeer
                    if (succ in self.frontier):
                        cur_front_cost = self.frontier.get(succ)
                        #if the current cost in the frontier than it would be if we took this alternative path for succ, replace cur frontier entry for succ with a new cost one
                        if (cur_front_cost > alt_cost):
                            self.parents[succ] = node
                            self.frontier.remove(succ)
                            self.frontier.add(succ, self.get_cost(succ))


                    #successor is not in frontier and has not been explored, so this is the first time seeing it, add it to the frontier
                    else:
                        self.parents[succ] = node
                        self.frontier_count += 1
                        self.frontier.add(succ, self.get_cost(succ))

        # if we get here, the search failed
        return self.get_results_dict(None) 

    def _transition_cost(self, state, move):
        #first find where the 0 is, bottom left of board is x = 0, y = 0
        x = state.find("0")[0]
        y = state.find("0")[1]
        if move == "up":
            #tile to be moved is under the 0
            return int(state.get_tile(x, y-1))**2
        elif move == "down": 
            #tile to be moved is above the 0
            return int(state.get_tile(x, y+1))**2
        elif move == "left":
            #tile to be moved is to the right of the 0
            return int(state.get_tile(x+1, y))**2
        elif move == "right": 
            #tile to be moved is to the left of the the 0
            return int(state.get_tile(x-1, y))**2
        else:
            print("error in _transition_cost")
        
        
class GreedySolver(UniformCostSolver):
    def __init__(self):
        super().__init__()
        self.frontier = pdqpq.PriorityQueue()

    def solve(self, start_state, heur):
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():

            node = self.frontier.pop()  # get the next node in the frontier queue
            #successor states to current node, also adds node to list of explored states
            succs = self.expand_node(node)
            if node == self.goal:
                return self.get_results_dict(node)

            #move is a stirng which is either left, right, up, or down, successor is the (max 4) successor boards for those 4 moves
            for move, succ in succs.items():
                #maybe take out
                if (not succ in self.frontier) and (not succ in self.explored):    
                    heuristic = self._heuristic(succ, heur)
                    self.parents[succ] = node
                    self.frontier_count += 1
                    self.frontier.add(succ, heuristic)        

        return self.get_results_dict(None) 

    def _heuristic(self, trans, heur):
        #misplaced tiles
        if (heur == "h1"):
            return self._num_misplaced_tiles(trans)
        elif (heur == "h2"):
            return self._manhattan_distance(trans)
        elif (heur == "h3"):
            return self._worse_manhattan(trans)
        else:
            print("error, heur was invalid in A*")

    def _num_misplaced_tiles(self, state):  
        num_misplaced_tiles = 0
        for x in range(0,3):
            for y in range(0,3):
                num_tile = int(state.get_tile(x,y))
                if (num_tile != 0) and (state.get_tile(x,y) != GOAL_STATE.get_tile(x,y)):
                    num_misplaced_tiles += 1
        return num_misplaced_tiles

    def _manhattan_distance(self, state):
        manhattan_distance = 0
        for x in range(0,3):
            for y in range(0,3):
                cur_tile = state.get_tile(x,y)
                x_coords = GOAL_STATE.find(cur_tile)[0]
                y_coords = GOAL_STATE.find(cur_tile)[1]
                num_tile = int(cur_tile)
                if num_tile != 0:
                    manhattan_distance += abs(x - x_coords) + abs(y - y_coords)
        return manhattan_distance

    def _worse_manhattan(self, state):
        manhattan_distance = 0
        for x in range(0,3):
            for y in range(0,3):
                cur_tile = state.get_tile(x,y)
                x_coords = GOAL_STATE.find(cur_tile)[0]
                y_coords = GOAL_STATE.find(cur_tile)[1]
                diff = abs(x - x_coords) + abs(y - y_coords)
                num_tile = int(cur_tile)
                manhattan_distance += num_tile**2 * diff
        return manhattan_distance


class AStarSolver(GreedySolver):
    def __init__(self):
        super().__init__()
        self.frontier = pdqpq.PriorityQueue()

    def solve(self, start_state, heur):
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():    

            node = self.frontier.pop()
            succs = self.expand_node(node)
            if node == self.goal:
                return self.get_results_dict(node)

            #move is a stirng which is either left, right, up, or down, successor is the (max 4) successor boards for those 4 moves
            for move, succ in succs.items(): 
                #node has not been explored
                if not succ in self.explored:
                    #successor is in frontier
                    if succ in self.frontier:
                        #heuristic to succ
                        succ_heuristic = self._heuristic(succ, heur)
                        #cost of moving from from node to succ
                        trans_to_succ = self._transition_cost(node, move)
                        #alternate priority in the queue
                        alt_priority = self.get_cost(node) + succ_heuristic + trans_to_succ
                        if (alt_priority < self.frontier.get(succ)):
                            self.parents[succ] = node
                            self.frontier.remove(succ)
                            self.frontier.add(succ, alt_priority)

                    #otherwise successor is not in frontier, first time seeing it
                    else:
                        self.parents[succ] = node
                        cost_to_succ = self.get_cost(succ)
                        succ_heuristic = self._heuristic(succ, heur)
                        succ_priority = cost_to_succ + succ_heuristic
                        self.frontier_count += 1
                        self.frontier.add(succ, succ_priority)

                #node has already been explored
                else:
                    continue

        return self.get_results_dict(None) 

    
    

def print_table(flav__results, include_path=False):
    """Print out a comparison of search strategy results.

    Args:
        flav__results (dictionary): a dictionary mapping search flavor tags result statistics. See
            solve_puzzle() for detail.
        include_path (bool): indicates whether to include the actual solution paths in the table

    """
    result_tups = sorted(flav__results.items())
    c = len(result_tups)
    na = "{:>12}".format("n/a")
    rows = [  # abandon all hope ye who try to modify the table formatting code...
        "flavor  " + "".join([ "{:>12}".format(tag) for tag, _ in result_tups]),
        "--------" + ("  " + "-"*10)*c,
        "length  " + "".join([ "{:>12}".format(len(res['path'])) if 'path' in res else na 
                                for _, res in result_tups ]),
        "cost    " + "".join([ "{:>12,}".format(res['path_cost']) if 'path_cost' in res else na 
                                for _, res in result_tups ]),
        "frontier" + ("{:>12,}" * c).format(*[res['frontier_count'] for _, res in result_tups]),
        "expanded" + ("{:>12,}" * c).format(*[res['expanded_count'] for _, res in result_tups])
    ]
    if include_path:
        rows.append("path")
        longest_path = max([ len(res['path']) for _, res in result_tups if 'path' in res ] + [0])
        print("longest", longest_path)
        for i in range(longest_path):
            row = "        "
            for _, res in result_tups:
                if len(res.get('path', [])) > i:
                    move, state = res['path'][i]
                    row += " " + move[0] + " " + str(state)
                else:
                    row += " "*12
            rows.append(row)
    print("\n" + "\n".join(rows), "\n")


def get_test_puzzles():
    """Return sample start states for testing the search strategies.
    
    Returns:
        A tuple containing three EightPuzzleBoard objects representing start states that have an
        optimal solution path length of 3-5, 10-15, and >=25 respectively.
    
    """ 
    # Note: test cases can be hardcoded, and are not required to be programmatically generated.
    #
    # fill in function body here
    #    
    small_length = puzz.EightPuzzleBoard("142375608")
    med_length = puzz.EightPuzzleBoard("314072685")
    long_boy = puzz.EightPuzzleBoard("802356174")
    return (small_length, med_length, long_boy)  # fix this line!


############################################

if __name__ == '__main__':

    # parse the command line args
    start = puzz.EightPuzzleBoard(sys.argv[1])
    if sys.argv[2] == 'all':
        flavors = ['bfs', 'ucost', 'greedy-h1', 'greedy-h2', 
                   'greedy-h3', 'astar-h1', 'astar-h2', 'astar-h3']
    else:
        flavors = sys.argv[2:]

    # run the search(es)
    results = {}
    for flav in flavors:
        print("solving puzzle {} with {}".format(start, flav))
        results[flav] = solve_puzzle(start, flav)

    print_table(results, include_path=False)  # change to True to see the paths!



import numpy as np
from microchannels import Microchannel
from materials import Silicon, Air
from customhys import hyperheuristic as hh

class MicrochannelDesign():
    def __init__(self):
        self.model = Microchannel(base=Silicon(), coolant=Air())
        #self.max_search_range = np.array([self.model.W_d/10, self.model.W_d/10, 0.005])
        #self.min_search_range = np.array([1e-24, 1e-12, 1e-8])
        self.max_search_range = np.array([2.72e-4, 8.5e-4, 1e-2])
        self.min_search_range = np.array([4.08e-12, 8.5e-12, 1e-6])
        #self.func_name = 'Microchannel entropy generation model'
        #self.max_search_range = np.array([2.65e-5, 20.4e-5])
        #self.min_search_range = np.array([4.53e-5, 13.6e-5])

    def get_func_val(self, variables):
        self.model.w_w = variables[0]
        self.model.w_c = variables[1]
        self.model.G_d = 0.007
        #print(self.model.G_d)
        #print(self.model.alpha)
        #print(self.model.beta)
        return self.model.sgen

    def get_formatted_problem(self, is_constrained=True):
        return dict(function=self.get_func_val,
                    boundaries=(self.min_search_range, self.max_search_range),
                    is_constrained=is_constrained)

fun = MicrochannelDesign()

with open('./collections/' + 'default.txt', 'r') as operators_file:
    heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]

prob = fun.get_formatted_problem(is_constrained=True)
hyp = None

for i in range(1,3):
    del hyp
    hyp = hh.Hyperheuristic(heuristic_space=heuristic_space, problem=prob, file_label="Test_%d"%i, parameters=
                            dict(cardinality=3,                # Max. numb. of SOs in MHs, lvl:1
                              num_iterations=100,           # Iterations a MH performs, lvl:1
                              num_agents=30,                # Agents in population,     lvl:1
                              num_replicas=30,              # Replicas per each MH,     lvl:2
                              num_steps=100,                # Trials per HH step,       lvl:2
                              stagnation_percentage=0.3,    # Stagnation percentage,    lvl:2
                              max_temperature=200,          # Initial temperature (SA), lvl:2
                              cooling_rate=0.1)
    )
    sol, perf, e_sol = hyp.run()
    print( sol, perf, e_sol, "\n\n")
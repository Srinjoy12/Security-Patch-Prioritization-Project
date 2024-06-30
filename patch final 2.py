import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go


factors = [
    "Criticality", "CVSS Score", "Business Functionality", "User Impact",
    "Active Exploits", "Exploitability", "Legal Obligations", "Audits",
    "Public-Facing Systems", "Risk Assessment", "Patch Stability", "Compatibility",
    "Previous Incidents"
]

real_life_incidents = [
    {"Criticality": 7, "CVSS Score": 9, "Business Functionality": 6, "User Impact": 8, "Active Exploits": 7, "Exploitability": 8, "Legal Obligations": 6, "Audits": 7, "Public-Facing Systems": 8, "Risk Assessment": 7, "Patch Stability": 6, "Compatibility": 5, "Previous Incidents": 7, "Impact": 8},
    {"Criticality": 5, "CVSS Score": 6, "Business Functionality": 4, "User Impact": 5, "Active Exploits": 6, "Exploitability": 5, "Legal Obligations": 4, "Audits": 5, "Public-Facing Systems": 6, "Risk Assessment": 5, "Patch Stability": 4, "Compatibility": 6, "Previous Incidents": 5, "Impact": 5},
    {"Criticality": 6, "CVSS Score": 8, "Business Functionality": 5, "User Impact": 7, "Active Exploits": 5, "Exploitability": 6, "Legal Obligations": 5, "Audits": 6, "Public-Facing Systems": 7, "Risk Assessment": 6, "Patch Stability": 5, "Compatibility": 7, "Previous Incidents": 6, "Impact": 7},
    {"Criticality": 8, "CVSS Score": 7, "Business Functionality": 7, "User Impact": 6, "Active Exploits": 8, "Exploitability": 7, "Legal Obligations": 7, "Audits": 8, "Public-Facing Systems": 5, "Risk Assessment": 8, "Patch Stability": 7, "Compatibility": 8, "Previous Incidents": 7, "Impact": 9},
    {"Criticality": 4, "CVSS Score": 5, "Business Functionality": 3, "User Impact": 4, "Active Exploits": 4, "Exploitability": 4, "Legal Obligations": 3, "Audits": 4, "Public-Facing Systems": 4, "Risk Assessment": 3, "Patch Stability": 3, "Compatibility": 4, "Previous Incidents": 4, "Impact": 4},
    {"Criticality": 9, "CVSS Score": 9, "Business Functionality": 9, "User Impact": 9, "Active Exploits": 9, "Exploitability": 9, "Legal Obligations": 9, "Audits": 9, "Public-Facing Systems": 9, "Risk Assessment": 9, "Patch Stability": 9, "Compatibility": 9, "Previous Incidents": 9, "Impact": 10},
    {"Criticality": 3, "CVSS Score": 4, "Business Functionality": 2, "User Impact": 3, "Active Exploits": 3, "Exploitability": 3, "Legal Obligations": 2, "Audits": 3, "Public-Facing Systems": 2, "Risk Assessment": 3, "Patch Stability": 2, "Compatibility": 3, "Previous Incidents": 3, "Impact": 3},
    {"Criticality": 8, "CVSS Score": 8, "Business Functionality": 8, "User Impact": 8, "Active Exploits": 8, "Exploitability": 8, "Legal Obligations": 8, "Audits": 8, "Public-Facing Systems": 8, "Risk Assessment": 8, "Patch Stability": 8, "Compatibility": 8, "Previous Incidents": 8, "Impact": 9},
    {"Criticality": 7, "CVSS Score": 7, "Business Functionality": 7, "User Impact": 7, "Active Exploits": 7, "Exploitability": 7, "Legal Obligations": 7, "Audits": 7, "Public-Facing Systems": 7, "Risk Assessment": 7, "Patch Stability": 7, "Compatibility": 7, "Previous Incidents": 7, "Impact": 8},
    {"Criticality": 6, "CVSS Score": 6, "Business Functionality": 6, "User Impact": 6, "Active Exploits": 6, "Exploitability": 6, "Legal Obligations": 6, "Audits": 6, "Public-Facing Systems": 6, "Risk Assessment": 6, "Patch Stability": 6, "Compatibility": 6, "Previous Incidents": 6, "Impact": 7}
]

historical_incidents = real_life_incidents

for incident in historical_incidents:
    incident["Impact"] = np.random.randint(1, 10)


factor_scales = {factor: 10 for factor in factors}


population_size = 100
num_generations = 100
mutation_rate = 0.1
num_parents_mating = 20


def evaluate_weights(weights):
    total_score = 0
    for incident in historical_incidents:
        predicted_impact = np.dot(weights, [incident[factor] / factor_scales[factor] for factor in factors])
        actual_impact = incident["Impact"]
        total_score += (predicted_impact - actual_impact) ** 2
    return -total_score  # Return negative score because we want to minimize MSE


def generate_population(size, num_factors):
    population = np.zeros((size, num_factors))
    for i in range(size):
        individual = np.random.dirichlet(np.ones(num_factors), size=1)[0]
        population[i, :] = individual
    return population

def select_mating_pool(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size, num_factors):
    offspring = np.empty((offspring_size, num_factors))
    crossover_point = np.uint8(num_factors * np.random.uniform(low=0.0, high=1.0, size=offspring_size))
    for k in range(offspring_size):
        offspring[k, 0:crossover_point[k]] = parents[np.random.randint(0, parents.shape[0], 1), 0:crossover_point[k]]
        offspring[k, crossover_point[k]:] = parents[np.random.randint(0, parents.shape[0], 1), crossover_point[k]:]
    return offspring


def mutate(offspring_crossover, mutation_rate):
    for idx in range(offspring_crossover.shape[0]):
        if np.random.rand() < mutation_rate:
            random_index = np.random.randint(0, offspring_crossover.shape[1])
            offspring_crossover[idx, random_index] = np.random.rand()
            offspring_crossover[idx] = offspring_crossover[idx] / np.sum(offspring_crossover[idx])  # Normalize to sum to 1
    return offspring_crossover
def calculate_patch_score(solution, factor_scales, patch_values):
    patch_scores = []
    for patch in patch_values:
        score = np.dot(solution, [patch[i] / factor_scales[factors[i]] for i in range(len(factors))])
        patch_scores.append(score)
    return patch_scores

# Main GA function
def genetic_algorithm(population_size, num_generations, num_parents_mating, mutation_rate):
    num_factors = len(factors)
    population = generate_population(population_size, num_factors)
    best_outputs = []
    for generation in range(num_generations):
        fitness = [evaluate_weights(individual) for individual in population]
        parents = select_mating_pool(population, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, population_size - num_parents_mating, num_factors)
        offspring_mutation = mutate(offspring_crossover, mutation_rate)
        population[:num_parents_mating] = parents
        population[num_parents_mating:] = offspring_mutation
        best_outputs.append(max(fitness))
    best_solution_idx = np.argmax(fitness)
    best_solution = population[best_solution_idx]
    return best_solution

best_solution = genetic_algorithm(population_size, num_generations, num_parents_mating, mutation_rate)
print("best_solution:", best_solution)

# Dash app setup
app = dash.Dash(__name__)

def add_patch_row(n_patches):
    return html.Tr([
        html.Td(f"Patch {n_patches+1}"),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-criticality", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-cvss", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-business-functionality", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-user-impact", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-active-exploits", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-exploitability", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-legal-obligations", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-audits", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-public-facing-systems", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-risk-assessment", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-patch-stability", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-compatibility", type='number', value=0)),
        html.Td(dcc.Input(id=f"patch-{n_patches+1}-previous-incidents", type='number', value=0)),
    ])

app.layout = html.Div([
    html.H1("Patch Priority Calculator"),
    html.P("Enter the number of patches:"),
    dcc.Input(id="num-patches", type="number", value=1),
    html.Button("Add Patches", id="add-patches", n_clicks=0),
    html.Table([
        html.Thead(html.Tr([
            html.Th("Patch"), html.Th("Criticality"), html.Th("CVSS Score"), html.Th("Business Functionality"),
            html.Th("User Impact"), html.Th("Active Exploits"), html.Th("Exploitability"), html.Th("Legal Obligations"),
            html.Th("Audits"), html.Th("Public-Facing Systems"), html.Th("Risk Assessment"), html.Th("Patch Stability"),
            html.Th("Compatibility"), html.Th("Previous Incidents")
        ])),
        html.Tbody(id="patch-table")
    ]),
    html.Button("Calculate Patch Priority", id="calculate-priority", n_clicks=0),
    html.Div(id="patch-priority-output")
])

@app.callback(
    Output("patch-table", "children"),
    Input("add-patches", "n_clicks"),
    State("num-patches", "value")
)
def add_patches(n_clicks, num_patches):
    return [add_patch_row(i) for i in range(num_patches)]

@app.callback(
    Output("patch-priority-output", "children"),
    Input("calculate-priority", "n_clicks"),
    State("patch-table", "children")
)
def calculate_patch_priority(n_clicks, patch_table):
    if patch_table is None:
        return "No patches to evaluate"
    
    patch_values = []
    for row in patch_table:
        if isinstance(row, dict) and row.get('type') == 'Tr':
            patch_data = []
            for cell in row['props']['children'][1:]:
                if isinstance(cell, dict) and 'props' in cell and 'children' in cell['props']:
                    value = cell['props']['children']['props']['value']
                    try:
                        patch_data.append(float(value))
                    except (ValueError, TypeError):
                        patch_data.append(0)  
                else:
                    patch_data.append(0)  
            patch_values.append(patch_data)
    
    print("patch_table:", patch_table)  
    print("patch_values:", patch_values)  
    
    
    best_solution = genetic_algorithm(population_size, num_generations, num_parents_mating, mutation_rate)
    patch_scores = calculate_patch_score(best_solution, factor_scales, patch_values)
    
    print("patch_scores:", patch_scores)  
    
    # Create a graph representation of the patch scores
    fig = go.Figure(data=go.Bar(x=[f"Patch {i+1}" for i in range(len(patch_scores))], y=patch_scores))
    fig.update_layout(title="Patch Priority Scores", xaxis_title="Patches", yaxis_title="Score")
    graph = dcc.Graph(figure=fig)
    
 
    return html.Div([graph])



if __name__ == "__main__":
    app.run_server(debug=True)
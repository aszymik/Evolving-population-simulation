from simulation import Population
import pandas as pd
import plotly.express as px


# plot 1: env_change, mutation_prob, number of organisms
def generate_plot_data(env_changes, mutation_probs, generations, T):
    df = pd.DataFrame(columns=['env_change', 'mutation_prob', 'population'])

    for env_change in env_changes:
        for mutation_prob in mutation_probs:
            population = Population(N=200,
                                    max_N=1000,
                                    n=2,
                                    env_change=env_change,
                                    env_mode='Global',
                                    T=T,
                                    mutation_prob=mutation_prob,
                                    mutation_std=0.3,
                                    fitness_std=0.2,
                                    fitness_thr=0.3,
                                    max_num_children=4,
                                    angle=30)
            _, optimum_data = population.simulation(generations + 5*10)
            tmp = 0
            for opt in range(len(optimum_data)):
                if opt*T + generations > optimum_data.shape[1]:
                    break
                tmp += optimum_data[opt, opt*T + generations]
            tmp /= len(optimum_data)
            df.loc[len(df)] = [env_change, mutation_prob, tmp]

    return df


env_changes = [e/100 for e in range(1, 11,)]
mutation_probs = [e/10 for e in range(1, 11,)]

generations = 20
T = 5

plot_data = generate_plot_data(env_changes, mutation_probs, generations, T)
plot_data.to_csv('plot_data.csv')

fig = px.density_heatmap(plot_data, x="mutation_prob", y="env_change", z="population",
                         histfunc="avg",
                         nbinsx=10, nbinsy=10)
fig.write_image("graphics/envs_plot.png")
fig.show()

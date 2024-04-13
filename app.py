import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from simulation import Population


def main():
    st.title("Population Simulation")

    with st.sidebar:
        st.title("Select parameters")
        generations = st.number_input("Number of generations", value=50)
        N = st.number_input("Initial population size", value=200)
        max_N = st.number_input("Maximum population size", value=1000)
        # n = st.number_input("Number of genes", value=2)
        env_change = st.number_input("Environmental change rate", value=0.02)
        T = st.number_input("Time for significant environmental change", value=50)
        mutation_prob = st.slider("Mutation probability", 0.0, 1.0, 0.75)
        mutation_std = st.number_input("Mutation standard deviation", value=0.3)
        fitness_std = st.number_input("Fitness standard deviation", value=0.2)
        reproduction_thr = st.slider("Reproduction threshold", 0.0, 1.0, 0.5)
        max_num_children = st.number_input("Maximum number of children", value=7)
        angle = st.number_input("Rotation angle of optimal genotypes", value=30)

    if st.button("Run Simulation"):
        population = Population(N=N,
                                max_N=max_N,
                                n=2,
                                env_change=env_change,
                                T=T,
                                mutation_prob=mutation_prob,
                                mutation_std=mutation_std,
                                fitness_std=fitness_std,
                                reproduction_thr=reproduction_thr,
                                max_num_children=max_num_children,
                                angle=angle)

        df = population.simulation(generations=generations)
        df.to_csv('df_all.csv')
        df = pd.read_csv('df_all.csv')
        opacity = np.where(df['type'] == 'organism', 1, 0.1)  # 1 for population, 0.5 for optimum

        col = px.colors.qualitative.Pastel
        fig = px.scatter(df,
                         x="x", 
                         y="y",
                         size="radius",
                         opacity=opacity,
                         animation_frame="generation",
                         color="type",
                         color_discrete_sequence=[col[9], col[7]],
                         )

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30  # ms
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
import streamlit as st
import plotly.express as px
from simulation import *


def main():
    st.title("Population Simulation")

    with st.sidebar:
        st.title("Select parameters")
        N = st.number_input("Initial population size", value=200)
        max_N = st.number_input("Maximum population size", value=1000)
        n = st.number_input("Number of genes", value=2)
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
                                n=n,
                                env_change=env_change,
                                T=T,
                                mutation_prob=mutation_prob,
                                mutation_std=mutation_std,
                                fitness_std=fitness_std,
                                reproduction_thr=reproduction_thr,
                                max_num_children=max_num_children,
                                angle=angle)
        df_population, df_optimum = population.simulation(generations=155)

        # kolumny x, y, generation
        fig = px.scatter(df_population, 
                         x="x", 
                         y="y", 
                         size="radius",
                         animation_frame="generation")
        
        fig.add_scatter(df_optimum,

        )


if __name__ == "__main__":
    main()

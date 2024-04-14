import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from simulation import Population


def main():
    st.set_page_config(page_title="Population Simulation",
                       page_icon=":whale:")
    st.title("Population Simulation")

    with st.sidebar:
        st.title("Select parameters")
        generations = st.number_input("Number of generations", value=30)
        N = st.number_input("Initial population size", value=200)
        max_N = st.number_input("Maximum population size", value=1000)
        # n = st.number_input("Number of genes", value=2)
        env_change = st.number_input("Environmental change rate", value=0.1)
        T = st.number_input("Time for significant environmental change", value=5)
        mutation_prob = st.slider("Mutation probability", 0.0, 1.0, 0.75)
        mutation_std = st.number_input("Mutation standard deviation", value=0.3)
        fitness_std = st.number_input("Fitness standard deviation", value=0.2)
        fitness_thr = st.slider("Fitness threshold for survival", 0.0, 1.0, 0.3)
        max_num_children = st.number_input("Maximum number of children", value=7)
        angle = st.number_input(u"Rotation angle of optimal genotypes (\xb0)", value=30)

    if st.button("Run Simulation"):
        population = Population(N=N,
                                max_N=max_N,
                                n=2,
                                env_change=env_change,
                                T=T,
                                mutation_prob=mutation_prob,
                                mutation_std=mutation_std,
                                fitness_std=fitness_std,
                                fitness_thr=fitness_thr,
                                max_num_children=max_num_children,
                                angle=angle)

        # df = population.simulation(generations=generations)
        # df.to_csv('df_all.csv')
        df = pd.read_csv('df_all.csv')

        # Calculate the number of individuals per generation
        df_organisms = df[df['type'] == 'organism']
        num_individuals = df_organisms.groupby('generation').size()

        fig = px.scatter(df,
                         x="x", 
                         y="y",
                         size="radius",
                         animation_frame="generation",
                         color="type",
                         color_discrete_map={'optimum':'rgba(201, 219, 116, 0.2)', 'organism':'rgba(180, 151, 231, 1.0)'},
                         title=f"Population size: {num_individuals[0]}"
                         )

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30  # ms
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
  
        # Change frame titles
        for button in fig.layout.updatemenus[0].buttons:
            button['args'][1]['frame']['redraw'] = True

        for i in range(len(fig.frames)):
            fig.frames[i]['layout'].update(title_text=f'<b>Population size: {num_individuals[i]}</b>')

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
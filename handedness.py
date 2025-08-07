import numpy as np
import preliz as pz
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# This is a Streamlit app for visualizing the distribution of handedness in a finite population.
# This is inspired by the app https://huggingface.co/spaces/Madhav/Handedness

pio.templates["custom"] = pio.templates["plotly"]
pio.templates.default = "custom"
# Set global number format (this affects tick labels, not tooltips directly)
pio.templates["custom"].layout.xaxis.tickformat = '.2f'
pio.templates["custom"].layout.yaxis.tickformat = '.2f'

st.set_page_config(layout="wide")

COLOR = "#36acc6"

def plot_distributions(alpha, beta, left_count, right_count, num_people):
    prior = pz.Beta(alpha, beta)
    x = np.linspace(0, 1, 1000)

    posterior_left = pz.Beta(alpha + left_count, beta + right_count)
    likelihood = pz.Beta(left_count+1, right_count+1)
    
    total_count = left_count + right_count
    maximum_likelihood_estimate = 0 if total_count == 0 else left_count / total_count

    x_vals = np.arange(0, num_people+1)
    posterior_mle = pz.Binomial(num_people - total_count, p=maximum_likelihood_estimate).pdf(x_vals)
    prior_pred = pz.BetaBinomial(alpha, beta, num_people).pdf(x_vals)
    posterior_pred = pz.BetaBinomial(alpha + left_count, beta + right_count, num_people - total_count).pdf(x_vals)

    fig = make_subplots(rows=2, cols=3, subplot_titles=[
        "<b>Prior</b>", "<b>Likelihood</b>", "<b>Posterior</b>",
        "<b>Prior Predictive Distribution</b>", "<b>Predictions given MLE</b>", "<b>Posterior Predictive Distribution</b>"
    ])

    # Prior
    fig.add_trace(go.Scatter(x=x, y=prior.pdf(x), mode="lines",
                             line=dict(color=COLOR, width=3)), row=1, col=1)

    if total_count > 0:
        # Likelihood
        fig.add_trace(go.Scatter(x=x, y=likelihood.pdf(x), mode="lines",
                                 line=dict(color=COLOR, width=3)), row=1, col=2)
        # Posterior
        fig.add_trace(go.Scatter(x=x, y=posterior_left.pdf(x), mode="lines",
                                line=dict(color=COLOR, width=3)), row=1, col=3)

    # Prior Predictive Distribution
    fig.add_trace(go.Bar(x=np.arange(0, len(prior_pred)), y=prior_pred,
                         marker_color=COLOR), row=2, col=1)

    
    if total_count > 0:
        # Predictions given MLE
        fig.add_trace(go.Bar(x=np.arange(left_count, num_people-total_count+left_count+1), y=posterior_mle,
                             marker_color=COLOR), row=2, col=2)
        
        # Posterior Predictive Distribution
        fig.add_trace(go.Bar(x=np.arange(left_count, num_people-total_count+left_count+1), y=posterior_pred,
                            marker_color=COLOR), row=2, col=3)

    fig.update_xaxes(range=[-0.5, num_people+0.5], row=2, col=2)
    fig.update_xaxes(range=[-0.5, num_people+0.5], row=2, col=3)

    fig.update_layout(
        height=800, width=1200,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode=False,
        font=dict(color="black"),
        autosize=False,
        margin=dict(l=20, r=20, t=20, b=20),

    )

    fig.update_xaxes(showgrid=False, tickfont=dict(color="black"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color="black"))
    for col in range(1, 4):  # columns 1, 2, 3
        fig.update_xaxes(title_text="Left handedness", row=1, col=col, title_font=dict(color="black"))
    for col in range(1, 4):  # columns 1, 2, 3
        fig.update_xaxes(title_text="Number of left handed students", row=2, col=col, title_font=dict(color="black"))

    st.plotly_chart(fig, use_container_width=False)

def main():
    st.markdown(
        """
        <div style="text-align:center">
        <h2 style="margin-top: -20px; margin-bottom: 0;">Handedness Analysis</h2>
        <h6 style="margin-top: -20px; margin-bottom: 20px;">Explore the distribution of handedness in a finite population</h6>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Parameter selection")
    if "num_people" not in st.session_state:
        st.session_state.num_people = 10
    
    alpha = st.sidebar.number_input('Alpha:', min_value=0.0, step=0.1, value=1.0)
    beta = st.sidebar.number_input('Beta:', min_value=0.0, step=0.1, value=1.0)
    left_count = st.sidebar.number_input('Left-handed count:', min_value=0, step=1)
    right_count = st.sidebar.number_input('Right-handed count:', min_value=0, step=1)

    total = left_count + right_count

    if total > st.session_state.num_people:
        st.session_state.num_people = total

    num_people = st.sidebar.number_input(
        'Total Population size:',
        min_value=1,
        step=1,
        key="num_people"
    )

    plot_distributions(alpha, beta, left_count, right_count, num_people)



if __name__ == '__main__':
    main()

import os, zipfile, secrets, shutil
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from streamlit_image_select import image_select
from impedance.models.circuits import CustomCircuit

st.set_page_config(layout="wide")


if "models" not in st.session_state:
    st.session_state["Token"] = secrets.token_hex(4)
    st.session_state["models"] = {}

models = st.session_state["models"]
token = st.session_state["Token"]


st.title("GES-EIS-ML-demo")
st.markdown(
    """
Welcome to the `GES-EIS-ML-demo` page! In this page you can play around with our machine learning
algorithm to recognize the equivalent circuit associated to a given Electrochemical
Impedance Spectroscopy (EIS) spectra.
"""
)


with st.form("ModelInputForm", clear_on_submit=True):

    st.markdown("### Model loading section")

    name = st.text_input("Set the name of the model", value=f"model_{len(models)}")
    zipdata = st.file_uploader("Load here the machine learning model in `.zip` format")

    submit = st.form_submit_button("📥 Add model")


if submit and zipdata != None:

    achive_name = zipdata.name.rstrip(".zip")

    os.mkdir(f"./tmp_{token}")
    try:
        with open(f"./tmp_{token}/model.zip", "wb") as file:
            file.write(zipdata.getvalue())

        with zipfile.ZipFile(f"./tmp_{token}/model.zip", "r") as file:
            file.extractall(f"./tmp_{token}/model")

        model = tf.keras.models.load_model(f"./tmp_{token}/model/{achive_name}")
        models[name] = model

    finally:
        shutil.rmtree(f"./tmp_{token}")


if models != {}:

    with st.expander("Input", expanded=True):
        st.markdown("## Input problem")

        img = image_select(
            "Select the equivalent circuit",
            ["img/0.png", "img/1.png", "img/2.png", "img/3.png"],
        )
        ID = int(img.rstrip(".png").lstrip("img/"))

        st.markdown("#### Settings")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("##### Series component")

            st.markdown("###### Resistor")
            R0 = st.number_input("Resistor (ohm)", min_value=0.01, max_value=2.0)

        with col2:

            st.markdown("##### First cell")

            st.markdown("###### Constant phase")
            CPE_1Q = st.number_input(
                "Q (Ohm^-1 sec^a)", min_value=0.01, max_value=2.0, key="CPEQ1"
            )
            CPE_1a = st.number_input("a", min_value=0.1, max_value=1.0, key="CPEa1")

            st.markdown("###### Resistor")
            R1 = st.number_input("R (ohm)", min_value=1.0, max_value=10.0, key="R1")

            if ID == 2 or ID == 3:
                st.markdown("###### Warburg impedance")
                W1 = st.number_input(
                    "W (Ohm sec^-1/2)", min_value=0.01, max_value=10.0, key="W1"
                )

        with col3:
            if ID == 1 or ID == 3:
                st.markdown("##### Second cell")

                st.markdown("###### Constant phase")
                CPE_2Q = st.number_input("Q (Ohm^-1 sec^a)", min_value=0.01, max_value=2.0)
                CPE_2a = st.number_input("a", min_value=0.1, max_value=1.0)

                st.markdown("###### Resistor")
                R2 = st.number_input("R (ohm)", min_value=1.0, max_value=10.0)

                if ID == 3:
                    st.markdown("###### Warburg impedance")
                    W2 = st.number_input("W (Ohm sec^-1/2)", min_value=0.01, max_value=10.0)

        FREQ_RANGE = [0.1, 1e6]
        FREQ_STEPS = 700

        orders = int(np.log10(FREQ_RANGE[1] / FREQ_RANGE[0]))
        delta_log = orders / (FREQ_STEPS - 1)

        frequency = np.array(
            [FREQ_RANGE[0] * 10 ** (n * delta_log) for n in range(FREQ_STEPS)]
        )

        if ID == 0:

            params = {"R0": R0, "CPE0_0": CPE_1Q, "CPE0_1": CPE_1a, "R1": R1}
            circuit = CustomCircuit(circuit="R0-p(CPE0,R1)", constants=params)

        elif ID == 1:

            params = {
                "R0": R0,
                "CPE0_0": CPE_1Q,
                "CPE0_1": CPE_1a,
                "R1": R1,
                "CPE1_0": CPE_2Q,
                "CPE1_1": CPE_2a,
                "R2": R2,
            }

            circuit = CustomCircuit(circuit="R0-p(CPE0,R1)-p(CPE1,R2)", constants=params)

        elif ID == 2:

            params = {"R0": R0, "CPE0_0": CPE_1Q, "CPE0_1": CPE_1a, "R1": R1, "W0": W1}

            circuit = CustomCircuit(circuit="R0-p(CPE0,R1-W0)", constants=params)

        else:

            params = {
                "R0": R0,
                "CPE0_0": CPE_1Q,
                "CPE0_1": CPE_1a,
                "R1": R1,
                "W0": W1,
                "CPE1_0": CPE_2Q,
                "CPE1_1": CPE_2a,
                "R2": R2,
                "W1": W2,
            }

            circuit = CustomCircuit(
                circuit="R0-p(CPE0,R1-W0)-p(CPE1,R2-W1)", constants=params
            )

        Z = circuit.predict(frequency, use_initial=True)

    st.markdown("#### EIS spectra")

    col1, col2 = st.columns(2)

    with col1:

        fig1, (ax1, ax2) = plt.subplots(nrows=2)

        ax1.semilogx(frequency, np.absolute(Z))
        ax2.semilogx(frequency, -(180 / np.pi) * np.angle(Z))

        ax1.set_ylabel(r"$|Z(\omega)| [\Omega]$", fontsize=16)
        ax1.grid(which="major", c="#DDDDDD")
        ax1.grid(which="minor", c="#EEEEEE")

        ax2.set_xlabel(r"$Frequency [Hz]$", fontsize=16)
        ax2.set_ylabel(r"$-\varphi(\omega) [deg]$", fontsize=16)
        ax2.grid(which="major", c="#DDDDDD")
        ax2.grid(which="minor", c="#EEEEEE")

        plt.tight_layout()

        st.pyplot(fig=fig1)

    with col2:

        fig2, ax3 = plt.subplots()

        ax3.plot(Z.real, -Z.imag)

        ax3.set_xlabel(r"Re(Z) $[\Omega]$", fontsize=16)
        ax3.set_ylabel(r"-Im(Z) $[\Omega]$", fontsize=16)
        ax3.grid(which="major", c="#DDDDDD")
        ax3.grid(which="minor", c="#EEEEEE")

        plt.tight_layout()

        st.pyplot(fig=fig2)

    with st.expander("Output", expanded=True):

        st.markdown("## 🤖 Machine-learning output")

        with st.spinner("Computing the prediction with various methods ..."):

            mloutput = {"Model": [], "Predicted ID": [], "Correctness": []}
            for name, model in models.items():

                X = np.concatenate((Z.real, Z.imag), axis=0)
                X = X.reshape([2, FREQ_STEPS])
                X = np.array([X])

                y = model.predict(X)
                y = tf.nn.softmax(y)
                y = np.array(y)

                predicted_ID = np.argmax(y)

                mloutput["Model"].append(name)
                mloutput["Predicted ID"].append(predicted_ID)
                mloutput["Correctness"].append("❌" if predicted_ID != ID else "✅")

        data = pd.DataFrame.from_dict(mloutput)
        st.table(data)

else:
    st.info("Please load at least one machine learning model to start the demo.")

# Titanic Survival Prediction App

## Overview

This is a Streamlit web application that predicts whether a passenger on the Titanic would have survived based on various input features. The model uses a logistic regression algorithm trained on the Titanic dataset.

## Features

- **Passenger Class (Pclass)**: The class of the ticket (1st, 2nd, or 3rd).
- **Sex**: The gender of the passenger (male or female).
- **Age**: The age of the passenger.
- **SibSp**: The number of siblings/spouses aboard.
- **Parch**: The number of parents/children aboard.
- **Fare**: The fare paid by the passenger.
- **Embarked**: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Requirements

To run this application, you need the following dependencies installed:

- Python 3.x
- Streamlit
- NumPy
- Pickle

You can install the required packages using pip:

```bash
pip install streamlit numpy
```

## Setup

1. **Download the Model and Encoder Files**: Ensure you have the `log_model.pkl`, `sex_encoder.pkl`, and `emb_encoder.pkl` files in the same directory as your script.
2. **Run the Application**: Use the following command to start the Streamlit server:

```bash
streamlit run TitanicApp.py
```

## Usage

1. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
2. Select the passenger features using the provided input fields.
3. Click the "Predict" button to see the survival prediction and probability.

## Example

- **Input**:
  - Passenger Class: 1
  - Sex: Female
  - Age: 25
  - SibSp: 1
  - Parch: 0
  - Fare: 72
  - Embarked: S

- **Output**:
  - Prediction: Survived
  - Probability: 0.85

## Contributing

Feel free to fork this repository and submit pull requests to improve the application. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

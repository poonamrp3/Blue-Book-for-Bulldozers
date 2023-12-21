from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from DataProcessing import cleanData 

app = Flask(__name__)

@app.route('/')
def index():
    # Call your cleanData function
    cleanData('data/Train.csv')

    # You can render an HTML template or return a string
    return "Data processing completed! Check the console for details."

if __name__ == '__main__':
    app.run(debug=True)

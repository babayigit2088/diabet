import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'diabet_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        s_test = flask.request.form['s_test']
        f_test = flask.request.form['f_test']
        l_test = flask.request.form['l_test']
        a_test = flask.request.form['a_test']
        b_test = flask.request.form['b_test']
        y_test = flask.request.form['y_test']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[s_test,f_test,l_test,a_test,b_test,y_test]],
                                       columns=['s_test', 'f_test', 'l_test','a_test','b_test','y_test'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'s_test':s_test,
                                                     'f_test':f_test,
                                                     'l_test':l_test,
                                                     'a_test':a_test,
                                                     'b_test':b_test,
                                                     'y_test':y_test},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
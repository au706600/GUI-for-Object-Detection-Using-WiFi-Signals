
import flask

import os

from Algorithm import show_objects

import numpy as np

from Algorithm_RIS import run_mse_esti

from al import statistical_model

app = flask.Flask(__name__) # flask instance

# store in dict and use for verification
user_credentials = {"Username": "Brian123", "Password": "JamesJac123"}

app.config['SECRET_KEY'] = os.urandom(12) # Generate secret key

@app.route('/')
def home(): # home page
    # Making sure, if not logged in, it redirects to the login page to avoid unauthorized access
    # by changing the route without login. 
    if "loginVerified" not in flask.session or not flask.session["loginVerified"]:
        return flask.redirect(flask.url_for('show_login'))
    else: 
        return flask.render_template('New_Page') 
    

@app.route('/login', methods = ['GET'])
def show_login(): # login page 
    return flask.render_template('login.html')
    

@app.route('/login', methods=['POST'])
def verification_login(): # verify login
    if flask.request.method == 'POST':
        # get username and password
        username = flask.request.form['Username']
        password = flask.request.form['Password']
        # verify username and password
        if username == user_credentials["Username"] and password == user_credentials["Password"]:
            flask.session["loginVerified"] = True
            return flask.redirect(flask.url_for('new_page'))
        else:
            flask.session["loginVerified"] = False
            return flask.render_template('login.html', error="Invalid credentials.")
    else:
        return flask.render_template('login.html')

@app.route('/New_Page', methods = ['GET', 'POST'])
def new_page(): # new page (main page) 

    # Making sure, if not logged in, it redirects to the login page to avoid unauthorized access 
    # by changing the route without login.
    if "loginVerified" not in flask.session or not flask.session["loginVerified"]:
        return flask.redirect(flask.url_for('show_login'))

    else:
        if 'coordinates' not in flask.session:
            flask.session['coordinates'] = []
        
        if 'esti_theta' not in flask.session:
            flask.session['esti_theta'] = []
        
        if 'esti_d' not in flask.session:
            flask.session['esti_d'] = []
        
        if 'collection' not in flask.session:
            flask.session['collection'] = []

            # Getting data x and y received in a flask request after posted form input
        x = flask.request.form.get('x', type = float)
        y = flask.request.form.get('y', type = float)
        if x is not None and y is not None:
            # Storing the coordinates in a list
            flask.session['coordinates'].append((x,y))

            RIS_pos = np.array([0, 4, 0])

            distance = np.sqrt((x - RIS_pos[0])**2+(y - RIS_pos[1])**2)

            theta_real = np.degrees(np.arctan2(y - RIS_pos[1], x - RIS_pos[0]))

            # Run MSE estimation
            esti_theta,esti_d = run_mse_esti(distance, theta_real, snrdb = 100, ietration = 10, risset = 0)

            flask.session['esti_theta'] = esti_theta.tolist()

            flask.session['esti_d'] = esti_d.tolist()   

            flask.session['collection'].append((esti_theta.tolist(), esti_d.tolist()))


            # Storing the estimation coordinates based on esti_theta and esti_d: 

            # Updating the session and storing the coordinates in a list in order to display, 
            # so it has to be stored, in order to continually input coordinates
            flask.session.modified = True


        coordinates = np.array(flask.session.get('coordinates', []))

        esti_theta = np.array(flask.session.get('esti_theta', []))

        esti_d = np.array(flask.session.get('esti_d', []))

        collection = flask.session.get('collection', [])

            # Showing the objects in the room
        output = show_objects(coordinates, esti_theta, esti_d, collection) 
        # Showing the statistical graphs to verify the estimation of objects
        st_output, st_output2 = statistical_model(coordinates, esti_theta, esti_d, collection)
        # returning the output of the show_objects function
        return flask.render_template('New_Page.html', output=output, st_output=st_output, st_output2=st_output2)

@app.route('/about')
def About(): # about page
    # Making sure, if not logged in, it redirects to the login page to avoid unauthorized access 
    # by changing the route without login.
    if "loginVerified" not in flask.session or not flask.session["loginVerified"]:
        return flask.redirect(flask.url_for('show_login'))
    else:
        return flask.render_template('about.html')

@app.route('/support')
def Support(): # support page
    # Making sure, if not logged in, it redirects to the login page to avoid unauthorized access 
    # by changing the route without login.
    if "loginVerified" not in flask.session or not flask.session["loginVerified"]:
        return flask.redirect(flask.url_for('show_login'))
    else:
        return flask.render_template('support.html')

if __name__ == '__main__':
    app.run(debug=True) # run flask instance











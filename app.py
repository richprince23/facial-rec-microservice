# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

from crypt import methods
from flask import Flask, request

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route('/recognize', methods =['POST'])
def recognize():
    file = request.files['image']
    
    return 


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()


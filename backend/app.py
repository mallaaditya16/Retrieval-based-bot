from flask import Flask
from flask import request
from flask_cors import CORS
from deployment import *
import pickle

app=Flask(__name__)
CORS(app)

file1=open("allresponses.pkl",'rb')
answers=pickle.load(file1)
file1.close()

@app.route('/api/',methods=["POST"])
def first():
    s=request.data
    s=s.decode("utf-8")
    #function imported from deployment.py
    return test(s,answers)